import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pcg.blocks import extract_blocks  # noqa: E402
from pcg.graph import build_graph, structural_importance  # noqa: E402
from pcg.evidence import collect_all  # noqa: E402
from pcg.inference import infer  # noqa: E402

src = open(os.path.join(PROJECT_ROOT, 'demo', 'candidate.py'), 'r', encoding='utf-8').read()
blocks = extract_blocks(src)
g = build_graph(blocks)
ev = collect_all(src, blocks, os.path.join(PROJECT_ROOT, 'demo', 'test_candidate.py'))
imp = structural_importance(g, blocks)
post = infer(blocks, g, ev, imp, damping=0.55)
print('Blocks:', len(blocks))
for bid, bp in post.items():
    print(bid, bp.posterior, bp.culpability, bp.inherited)
