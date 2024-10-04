#! C:\Users\vinay\Documents\workspace\Projects\ABSA-1\absa_env\Scripts\python.exe

from dotenv import load_dotenv
import ast
import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GenerationConfig
import torch
from peft import PeftModel
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import List

# # ... existing imports and code ...
# class Sentiment(BaseModel):
#     sentiment: str = Field(description="The sentiment of the aspect (positive, negative, or neutral)")

# class Aspect(BaseModel):
#     aspect: str = Field(description="The aspect of the review")

# class AspectList(BaseModel):
#     aspects: List[Aspect] = Field(description="List of comma separated aspects extracted from the review")


load_dotenv()

def load_models():
    """Load and return the specified model."""
    
    model_name='google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    peft_model_id="vbiramane/aspect_detection"
    peft_model = PeftModel.from_pretrained(original_model, peft_model_id, torch_dtype=torch.bfloat16, is_trainable=False)
    
    # pipe = pipeline("text2text-generation", model=peft_model, tokenizer=tokenizer,max_length=100)
    # peft_llm=HuggingFacePipeline(pipeline=pipe)
    
    pipe_original = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer,max_length=100)
    original_llm=HuggingFacePipeline(pipeline=pipe_original)
    
    return peft_model, original_llm, tokenizer

def load_sentiment_model():
    """Load and return the specified model."""
    
    model_name='google/flan-t5-base'
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    peft_model_id="vbiramane/absa"
    peft_model = PeftModel.from_pretrained(original_model, peft_model_id, torch_dtype=torch.bfloat16, is_trainable=False)
    
    pipe_original = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer,max_length=100)
    original_llm=HuggingFacePipeline(pipeline=pipe_original)
    
    return peft_model, original_llm, tokenizer

def peft_inference(peft_model, tokenizer, prompt):
    """Perform inference using a PEFT model."""
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # peft_model.to('cuda')
    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    
    return peft_model_text_output


def detect_aspects(context, review):
    """Detect aspects in a review given a context."""
    
    # parser = PydanticOutputParser(pydantic_object=AspectList)
    
    peft_model, original_llm, tokenizer = load_models()

    # Create LangChain components   
    prompt_template = """
    Provide the comma separated aspect terms related to {context} from below review in json format.

    {review}

    Aspect Terms: """

    prompt = PromptTemplate(template=prompt_template, input_variables=["review", ])

    chain_original = prompt | original_llm 

    # Use the fine-tuned LLM in your application
    aspects_peft = peft_inference(peft_model, tokenizer, prompt.invoke({"review": review, "context": context}).text)
    aspects_original = chain_original.invoke({"review": review, "context": context})

    return aspects_peft, aspects_original.split(',')

def detect_sentiment(aspect, review):
    """Detect sentiment of an aspect in a review."""
    
    peft_model, original_llm, tokenizer = load_sentiment_model()

    # Create LangChain components   
    prompt_template = """
        Analyze the following review and extract the sentiments of '{aspect}' aspect:

        Review: {review}
        
        """

    prompt = PromptTemplate(template=prompt_template, input_variables=["review", "aspect"])

    chain_original = prompt | original_llm 

    # Use the fine-tuned LLM in your application
    sentiment_peft = peft_inference(peft_model, tokenizer, prompt.invoke({"review": review, "aspect": aspect}).text)
    sentiment_original = chain_original.invoke({"review": review, "aspect": aspect})

    return sentiment_peft, sentiment_original
    
def detect_aspects_and_sentiment(context, review):
    """Detect aspects and sentiment in a review given a context."""
    results = []
    _, aspects_original = detect_aspects(context, review)
    for aspect in aspects_original:
        sentiment_peft, sentiment_original = detect_sentiment(aspect, review)
        print(f"Aspect: {aspect}")
        print(f"PEFT Sentiment: {sentiment_peft}")
        print(f"Original Sentiment: {sentiment_original}")
        
        results.append({
                'review': review,
                'aspect': aspect,
                'sentiment_peft': sentiment_peft,
                'sentiment_original': sentiment_original
            })
        
    return pd.DataFrame(results).groupby('review').agg({'aspect': lambda x: list(x), 'sentiment_peft': lambda x: list(x), 'sentiment_original': lambda x: list(x)}).reset_index()
        
if __name__ == "__main__":
    
    context_text = "restuarant"
    # review_text = "Yet paired with such rude service, would never recommend for anyone interested in carrying any kind of conversation while there."
    # review_text = "The food is very good for it's price, better than most fried dumplings I've had."
    # review_text = "If you live in Upper Manhattan, Siam Square is THE place for Thia food."
    # review_text = "Waitstaff is great, very attentive."
    # review_text = "The food is very good for it's price, but service is slow."
    true_aspects_sentiment_list = []
    peft_aspect_sentiment_list = []
    original_aspect_sentiment_list = []
    results = []
    
    df_in = pd.read_csv('absa_terms_polarity_test_set1.csv')
    print(df_in)
    for _, row in df_in.iloc[:2,:].iterrows():
        text = row['sentence']
        row_aspects = ast.literal_eval(row['term'])
        row_sentiments = ast.literal_eval(row['polarity'])
                
        true_aspects_sentiment = list([str(aspect).lower() + ":" + str(sentiment).lower() for aspect, sentiment in zip(row_aspects, row_sentiments)])
        df = detect_aspects_and_sentiment(context_text, text)
            
        peft_aspect_sentiment= list([str(aspect).lower() + ":" + str(sentiment).lower() for aspect, sentiment in zip(df['aspect'].values[0], df['sentiment_peft'].values[0])])
        original_aspect_sentiment= list([str(aspect).lower() + ":" + str(sentiment).lower() for aspect, sentiment in zip(df['aspect'].values[0], df['sentiment_original'].values[0])])
        
        df['true_aspects_sentiment'] = [true_aspects_sentiment]
        df['peft_model_aspect_sentiment'] = [peft_aspect_sentiment]
        df['original_model_aspect_sentiment'] = [original_aspect_sentiment]
        true_aspects_sentiment_list.append(true_aspects_sentiment)
        peft_aspect_sentiment_list.append(peft_aspect_sentiment)
        original_aspect_sentiment_list.append(original_aspect_sentiment)
        
        results.append(df.loc[:, ['review', 'true_aspects_sentiment', 'peft_model_aspect_sentiment', 'original_model_aspect_sentiment']])
        
    print(pd.concat(results))   
    
    rouge = evaluate.load('rouge')

    original_model_results = rouge.compute(
        predictions=[', '.join(x) for x in original_aspect_sentiment_list],
        references=[', '.join(x) for x in true_aspects_sentiment_list],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=[', '.join(x) for x in peft_aspect_sentiment_list],
        references=[', '.join(x) for x in true_aspects_sentiment_list],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)

    print('PEFT MODEL:')
    print(peft_model_results)
        