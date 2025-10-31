Check Yaml...............................................................Passed
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Passed
Flake8...................................................................Failed
- hook id: flake8
- exit code: 1

src/flag_gems/runtime/backend/_ascend/fused/fused_add_rms_norm.py:9:1: F401 'flag_gems.utils.triton_lang_extension as tle' imported but unused
src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py:11:1: F401 'flag_gems.utils.triton_lang_extension as tle' imported but unused
src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py:222:9: F841 local variable 'seq_len' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/__init__.py:64:1: F401 '.where.where_scalar_other' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/__init__.py:64:1: F401 '.where.where_scalar_self' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/__init__.py:64:1: F401 '.where.where_self' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/__init__.py:64:1: F401 '.where.where_self_out' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/arange.py:48:13: F821 undefined name 'runtime'
src/flag_gems/runtime/backend/_ascend/ops/gather.py:1:1: F401 'importlib' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:2:1: F401 'itertools' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:4:1: F401 'os' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:5:1: F401 'typing.Any' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:5:1: F401 'typing.Callable' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:5:1: F401 'typing.List' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:5:1: F401 'typing.Mapping' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:5:1: F401 'typing.Tuple' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:5:1: F401 'typing.Union' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:11:1: F401 'flag_gems.runtime' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:13:1: F401 'flag_gems.utils.broadcastable_to' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:14:1: F401 'flag_gems.utils.triton_lang_extension as tle' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:15:1: F401 'flag_gems.utils.code_cache.code_cache_dir' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:16:1: F401 'flag_gems.utils.code_utils.IndentedBuffer' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:16:1: F401 'flag_gems.utils.code_utils.write_atomic' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/gather.py:78:5: F841 local variable 'BLOCK_SIZE' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/gather.py:95:5: F841 local variable 'dim_stride' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/gather.py:96:5: F841 local variable 'inp_strided' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/gather.py:97:5: F841 local variable 'N' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/gather.py:99:18: F541 f-string is missing placeholders
src/flag_gems/runtime/backend/_ascend/ops/groupnorm.py:120:9: F841 local variable 'x_dtype' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/groupnorm.py:185:9: F841 local variable 'weight' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/groupnorm.py:190:9: F841 local variable 'bias' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/groupnorm.py:234:9: F841 local variable 'hw_size' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/index.py:4:1: F401 'numpy as np' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/index_select.py:24:9: E741 ambiguous variable name 'l'
src/flag_gems/runtime/backend/_ascend/ops/linspace.py:29:9: E741 ambiguous variable name 'l'
src/flag_gems/runtime/backend/_ascend/ops/linspace.py:65:9: F811 redefinition of unused 'grid' from line 63
src/flag_gems/runtime/backend/_ascend/ops/masked_fill.py:1:1: F401 'itertools' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/masked_fill.py:3:1: F401 'typing.List' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/masked_fill.py:3:1: F401 'typing.Tuple' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/masked_fill.py:3:1: F401 'typing.Union' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/repeat_interleave.py:5:1: F401 'triton.language as tl' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/repeat_interleave.py:7:1: F401 'flag_gems.utils.triton_lang_extension as tle' imported but unused
src/flag_gems/runtime/backend/_ascend/ops/slice_scatter.py:25:1: F811 redefinition of unused 'copy' from line 6
src/flag_gems/runtime/backend/_ascend/ops/vector_norm.py:69:5: F841 local variable 'base_offset' is assigned to but never used
src/flag_gems/runtime/backend/_ascend/ops/vector_norm.py:295:13: F841 local variable 'CORE_NUM' is assigned to but never used

clang-format.........................................(no files to check)Skipped
isort....................................................................Passed
black....................................................................Passed
black-jupyter............................................................Passed
