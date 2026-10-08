"""Fingerprint an existing local ONNX snapshot; never downloads model files."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_path',type=Path)
    parser.add_argument('--output',type=Path,default=Path('data/models/bge-reranker-base-onnx-manifest.json'))
    args=parser.parse_args()
    digests={}
    for name in ['config.json','tokenizer.json','tokenizer_config.json','onnx/model.onnx']:
        with (args.model_path/name).open('rb') as stream:
            digests[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'model':args.model_path.name,'sha256':digests},indent=2),encoding='utf8')
    print(args.output)


if __name__=='__main__':main()
