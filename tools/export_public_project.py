"""Export a small, inspectable project snapshot without local runtime assets."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
DIRECTORIES = ('gensim_lab', 'tools', 'tests', 'configs', 'docs', 'reports', 'figures', 'references')
ROOT_FILES = ('.gitignore', '.gitattributes', 'README.md', 'AGENTS.md', 'pyproject.toml', 'environment.yml',
              'requirements.lock.txt', 'requirements.paper.lock.txt')
FIXED_RECORDS = (
    'data/thucnews/prepared/9cb3ecd006ae1a9a1f7f/manifest.json',
    'artifacts/thucnews_validation/benchmarks/c1943509f229e0645ba2',
    'artifacts/thucnews_small_search/experiments/c0654ea5f83982f1f307e526/index.json',
)
EXTENSIONS = {'.py', '.pyx', '.ps1', '.yaml', '.yml', '.toml', '.md', '.csv', '.json',
              '.png', '.svg', '.txt', '.cc', '.h'}
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.git', '.artifact-work'}
TOKEN_PATTERN = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--refresh', action='store_true', help='Refresh selected files and the manifest in an existing export.')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if ROOT not in destination.parents or destination.parent.name != 'temp':
        raise ValueError('Destination must be a new directory under the project temp directory.')
    if destination.exists() and not args.refresh:
        raise FileExistsError('Export destination already exists; no files were overwritten.')
    selected = [ROOT / name for name in ROOT_FILES]
    excluded = []
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob('*')):
            if not path.is_file() or any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
                continue
            if path.relative_to(ROOT).as_posix() == 'docs/公开版本文件清单.md':
                continue
            if path.suffix.lower() not in EXTENSIONS and path.name != 'LICENSE':
                excluded.append(path.relative_to(ROOT).as_posix())
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                excluded.append(path.relative_to(ROOT).as_posix())
                continue
            selected.append(path)
    for name in FIXED_RECORDS:
        resource = ROOT / name
        paths = sorted(resource.rglob('*')) if resource.is_dir() else [resource]
        for path in paths:
            if path.is_file():
                assert path.suffix.lower() in {'.csv', '.json', '.txt'}
                assert path.stat().st_size <= MAX_FILE_BYTES
                selected.append(path)
    for path in selected:
        if path.suffix.lower() != '.png':
            text = path.read_text(encoding='utf-8-sig')
            if TOKEN_PATTERN.search(text):
                raise ValueError(f'Possible embedded credential or private key: {path.relative_to(ROOT)}')
    destination.mkdir(parents=True, exist_ok=args.refresh)
    sizes = []
    for path in sorted(selected):
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(path.read_bytes()).digest()
        sizes.append((relative.as_posix(), target.stat().st_size))
    total = sum(size for _, size in sizes)
    largest = max(sizes, key=lambda pair: pair[1])
    summary = ('# GitHub 公开版本文件清单\n\n'
               '仓库：[LynxPeng/Gensim-FastText](https://github.com/LynxPeng/Gensim-FastText)。首次提交信息：`init`。\n\n'
               f'本清单列出 {len(sizes)} 个项目文件，合计 {total / 1024 / 1024:.2f} MiB；每个文件均小于 5 MiB。另包含本文档。\n\n'
               '发布目录为源码、工具脚本、测试、配置、说明文档、报告、图表与附许可证的参考资源。另保留一份语料清洗清单、小型固定评价划分及候选选参记录；data 与 artifacts 中其余内容不导出。未导出新闻全文、临时目录、缓存、论文 PDF、压缩包、模型及编译工具。\n\n'
               '复制前扫描可能的凭据与私钥，复制后逐文件比对 SHA-256；上传不使用本地旧提交历史。\n\n'
               '| 文件 | 字节数 |\n| --- | ---: |\n')
    summary += ''.join(f'| `{name}` | {size:,} |\n' for name, size in sizes)
    (destination / 'docs/公开版本文件清单.md').write_text(summary, encoding='utf-8', newline='\n')
    print(f'Exported {len(sizes)} files; total {total / 1024 / 1024:.2f} MiB; largest {largest[0]} ({largest[1]} bytes).')
    print(f'Excluded unsupported/large reference or report files: {len(excluded)}.')
    print(f'Destination: {destination}')


if __name__ == '__main__':
    main()
