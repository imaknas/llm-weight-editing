"""IWIBench 公開回合的分析:各編輯模型實際寫了目標模型的哪些部位,與得分的關係。

資料來自 Hugging Face 資料集 hcltech-robotics/iwibench(family 0.2.2,330 個回合):
  <資料夾>/index.jsonl                       rollouts/index.jsonl
  <資料夾>/r/<trace_id>/patch.json            每回合的 patch 中繼資料
  <資料夾>/r/<trace_id>/service-trajectory.jsonl   每回合的工具呼叫紀錄(沒有寫入的回合沒有這個檔)
只用文字檔,不需要權重。用法:python analysis/iwibench_rollouts.py <資料夾>
"""
import ast, collections, json, os, re, statistics as st, sys

D = sys.argv[1]
idx = {}
for l in open(D + '/index.jsonl'):
    r = json.loads(l)
    if r['family_version'] == '0.2.2':
        idx[r['trace_id']] = r

def kind(name):
    if 'embed_tokens' in name or name.startswith('lm_head'): return 'embedding'
    m = re.search(r'layers\.(\d+)\.(.*)', name)
    if not m: return 'other:' + name
    rest = m.group(2)
    part = 'mlp' if 'mlp' in rest else 'attn' if 'attn' in rest else 'norm' if 'norm' in rest else rest
    return part

def layer(name):
    m = re.search(r'layers\.(\d+)\.', name)
    return int(m.group(1)) if m else None

recs = []
for tid, r in idx.items():
    src = r['published_source_row']
    src = ast.literal_eval(src) if isinstance(src, str) else src
    ed = r['editor_model'].split('/')[-1].split(':')[0]
    eff = r['reasoning_effort']
    label = ed + ('' if eff in (None, 'none') else f' ({eff})')
    rec = dict(tid=tid, editor=label, family=r['family'], reward=float(src['reward']), gain=float(src['gain']),
               retention=float(src['control_retention']), changed=int(float(src['changed_parameters'])),
               l2=float(src['patch_l2'] or 0), linf=float(src['patch_linf'] or 0))
    p = json.load(open(f'{D}/r/{tid}/patch.json'))
    sparse = json.loads(p.get('sparse') or '[]')
    rec['patch_tensors'] = p.get('tensors', 0)
    rec['patch_sparse_names'] = [s['name'] for s in sparse]
    writes, reads, inspects, runs = [], [], 0, 0
    f = f'{D}/r/{tid}/service-trajectory.jsonl'
    if os.path.exists(f):
        for l in open(f):
            t = json.loads(l)
            q = t.get('request', {})
            if t['tool'] == 'write_weights':
                writes.append((q.get('tensor', ''), q.get('operation'), t.get('ok'), q.get('selection')))
            elif t['tool'] == 'read_weights' and q.get('tensor'):
                reads.append(q['tensor'])
            elif t['tool'] == 'inspect_target': inspects += 1
            elif t['tool'] == 'run_target': runs += 1
    ok = [w for w in writes if w[2]]
    rec['n_writes'] = len(ok)
    rec['write_kinds'] = collections.Counter(kind(w[0]) for w in ok)
    rec['write_ops'] = collections.Counter(w[1] for w in ok)
    rec['write_layers'] = sorted({layer(w[0]) for w in ok if layer(w[0]) is not None})
    rec['rows'] = len({(w[0], w[3][0]) for w in ok if kind(w[0]) == 'embedding' and isinstance(w[3], list) and w[3] and isinstance(w[3][0], int)})
    rec['read_kinds'] = collections.Counter(kind(t) for t in reads)
    rec['inspects'], rec['runs'] = inspects, runs
    rec['patch_kinds'] = sorted({kind(n) for n in rec['patch_sparse_names']})
    recs.append(rec)

print('rollouts', len(recs))
tot = collections.Counter(); 
for r in recs: tot.update(r['write_kinds'])
print('\n成功的寫入,依模型部位:', dict(tot), ' 合計', sum(tot.values()))
ops = collections.Counter()
for r in recs: ops.update(r['write_ops'])
print('寫入操作:', dict(ops))
rk = collections.Counter()
for r in recs: rk.update(r['read_kinds'])
print('讀權重,依部位:', dict(rk))

def where(r):
    ks = set(r['write_kinds'])
    if not ks: return 'no write'
    if ks == {'embedding'}: return 'embedding only'
    if 'embedding' in ks: return 'embedding + internal'
    return 'internal only'
print('\n每回合改了哪裡 → 回合數、平均 reward、平均保留率')
g = collections.defaultdict(list)
for r in recs: g[where(r)].append(r)
for k, rs in g.items():
    print(f'  {k:22s} n={len(rs):3d} reward={st.mean(x["reward"] for x in rs):.3f} retention={st.mean(x["retention"] for x in rs):.3f}')

print('\n依編輯者:回合中改過內部層的比例 / 只改 embedding 的比例 / 沒寫入')
e = collections.defaultdict(list)
for r in recs: e[r['editor']].append(r)
for k, rs in sorted(e.items(), key=lambda kv: -st.mean(x['reward'] for x in kv[1])):
    c = collections.Counter(where(x) for x in rs)
    print(f'  {k:28s} reward={st.mean(x["reward"] for x in rs):.3f} emb_only={c["embedding only"]:2d} emb+int={c["embedding + internal"]:2d} int_only={c["internal only"]:2d} none={c["no write"]:2d} 讀內部層的回合={sum(1 for x in rs if set(x["read_kinds"])-{"embedding"}):2d} inspect中位數={st.median(x["inspects"] for x in rs)}')

print('\n依任務:改 embedding 列數中位數、改過內部層的回合數、平均 reward')
fm = collections.defaultdict(list)
for r in recs: fm[r['family']].append(r)
for k, rs in sorted(fm.items(), key=lambda kv: -st.mean(x['reward'] for x in kv[1])):
    c = collections.Counter(where(x) for x in rs)
    print(f'  {k:32s} reward={st.mean(x["reward"] for x in rs):.2f} rows_med={st.median(x["rows"] for x in rs):5.1f} internal={c["embedding + internal"]+c["internal only"]:2d}/{len(rs)} none={c["no write"]:2d} gain>0={sum(x["gain"]>0 for x in rs):2d} 有gain但保留率<0.95={sum(x["gain"]>0 and x["retention"]<0.95 for x in rs)}')

print('\n改過內部層的回合明細:')
for r in recs:
    if where(r) in ('embedding + internal', 'internal only'):
        print(f'  {r["editor"]:24s} {r["family"]:30s} reward={r["reward"]:.2f} ret={r["retention"]:.2f} kinds={dict(r["write_kinds"])} layers={r["write_layers"]} final_patch={r["patch_kinds"]}')

print('\n最終 patch 含哪些部位(依回合):', collections.Counter(tuple(r['patch_kinds']) for r in recs))
json.dump([{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in r.items()} for r in recs], open(D + '/analysis.json', 'w'))
