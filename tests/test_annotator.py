import pytest
from memory.annotator import AnnotationBatch, TimeAnnotator, LLMAnnotator
class FakeLLM:
    def __init__(self,result): self.result=result
    def complete(self,instruction,payload,schema): assert schema is AnnotationBatch; return self.result
def test_generic_grounded_offsets():
    result=AnnotationBatch(annotations=[dict(source_id='m1',text='Alice',label='person',confidence=.9)])
    out=LLMAnnotator({'person':'A person'},llm=FakeLLM(result)).annotate([{'source_id':'m1','content':'Alice moved to Paris.','timestamp':10}])
    assert out == [dict(source_id='m1',text='Alice',label='person',confidence=.9,start=0,end=5,source_timestamp=10)]
def test_time_keeps_expression():
    result=AnnotationBatch(annotations=[dict(source_id='m1',text='last Friday',label='relative_time',confidence=.8)])
    out=TimeAnnotator(FakeLLM(result)).annotate([{'source_id':'m1','content':'I ran last Friday.'}])
    assert out[0]['text']=='last Friday' and out[0]['start']==6 and out[0]['source_timestamp'] is None
@pytest.mark.parametrize('bad', [AnnotationBatch(annotations=[dict(source_id='other',text='Alice',label='person',confidence=1)]),AnnotationBatch(annotations=[dict(source_id='m1',text='Bob',label='person',confidence=1)]),AnnotationBatch(annotations=[dict(source_id='m1',text='Alice',label='unknown',confidence=1)])])
def test_rejects_ungrounded_output(bad):
    with pytest.raises(ValueError): LLMAnnotator({'person':'A person'},llm=FakeLLM(bad)).annotate([{'source_id':'m1','content':'Alice moved.'}])
