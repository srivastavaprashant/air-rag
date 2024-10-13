from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, GenerationConfig
from biokb.settings import DEVICE_NAME


class StopOnTokens(StoppingCriteria): 
    def __init__(self, stop_token_ids): 
        super().__init__()
        self.stop_token_ids = stop_token_ids
        self.token_len = len
    
    def __call__(self, input_ids, scores, **kwargs):        
        if (input_ids[0][-len(self.stop_token_ids)+1:] == self.stop_token_ids[1:]).all():
            return True
        return False

def get_stop_criteria(
    tokenizer: AutoTokenizer, 
    stop_words: list[str]
):
    stop_tokens = []
    for stop_word in stop_words:
        stop_token_ids=tokenizer.encode(
            stop_word, 
            return_tensors="pt", 
            add_special_tokens=False
        ).to(DEVICE_NAME).squeeze()

        stop_tokens.append(StopOnTokens(stop_token_ids))
    return StoppingCriteriaList(stop_tokens)

def get_generation_config(t=0.0,p=1.5):
    return GenerationConfig(
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=0,
        # eos_token_id=model.config.eos_token_id,
        # pad_token=model.config.pad_token_id,
        temperature=t,
        repetition_penalty=p
    )
