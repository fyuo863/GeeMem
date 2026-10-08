from types import SimpleNamespace
import numpy as np
import pytest
from memory.rerank import ONNXReranker


def fixture():
    r=ONNXReranker.__new__(ONNXReranker)
    r.batch=2;r.max_length=128;r.output='logits'
    r.inputs=[SimpleNamespace(name=n,type='tensor(int64)') for n in ['input_ids','attention_mask','token_type_ids']]
    calls=[]
    def tokenizer(queries,docs,**kwargs):
        assert kwargs['truncation'] and kwargs['max_length']==128
        return {'input_ids':np.array([[int(d)] for d in docs]),'attention_mask':np.ones((len(docs),1))}
    def run(outputs,feed):
        calls.append(feed)
        assert all(v.dtype==np.int64 for v in feed.values())
        assert not feed['token_type_ids'].any()
        return [feed['input_ids'].astype(float)]
    r.tokenizer=tokenizer;r.session=SimpleNamespace(run=run)
    return r,calls


def test_batches_preserve_order_and_raw_logits():
    r,calls=fixture()
    assert r.score('q',['-3','5','1']).tolist()==[-3,5,1]
    assert len(calls)==2
    assert r.score('q',[]).size==0 and len(calls)==2


@pytest.mark.parametrize('bad',[np.array([[float('nan')]]),np.array([[1,2]])])
def test_reject_invalid_model_output(bad):
    r,_=fixture();r.session.run=lambda *args:[bad]
    with pytest.raises(ValueError,match='scores'):r.score('q',['1'])
