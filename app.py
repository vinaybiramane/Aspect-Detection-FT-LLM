import streamlit as st
import evaluate
import pandas as pd
import ast
from aspects import detect_aspects_and_sentiment

st.title("Aspect-based Sentiment Analysis")
tab1, tab2 = st.tabs(["Single Review", "Bulk Analysis"])

with tab1:
    st.header("Analyze Single Review")
    review = st.text_area("Enter a review:")
    context = st.text_area("Enter the context:")
    
    if st.button("Analyze"):
        if review:
            st.write("Analyzing reviews...")
            # Single review input
            df = detect_aspects_and_sentiment(context, review)
            st.write("Analysis complete!")
            st.dataframe(df)
        else:
            st.warning("Please enter a review.")

with tab2:
    st.header("Bulk Analysis")   
# File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if st.button("Test File"):
        if uploaded_file is not None:
            df_in = pd.read_csv(uploaded_file)
            st.write("Analyzing reviews...")
            
            results = []
            true_aspects_sentiment_list = []
            peft_aspect_sentiment_list = []
            original_aspect_sentiment_list = []
            
            for _, row in df_in.iterrows():
                text = row['sentence']
                row_aspects = ast.literal_eval(row['term'])
                row_sentiments = ast.literal_eval(row['polarity'])
                
                true_aspects_sentiment = list([str(aspect).lower() + ":" + str(sentiment).lower() for aspect, sentiment in zip(row_aspects, row_sentiments)])
                df = detect_aspects_and_sentiment(context, text)
                
                peft_aspect_sentiment= list([str(aspect).lower() + ":" + str(sentiment).lower() for aspect, sentiment in zip(df['aspect'].values[0], df['sentiment_peft'].values[0])])
                original_aspect_sentiment= list([str(aspect).lower() + ":" + str(sentiment).lower() for aspect, sentiment in zip(df['aspect'].values[0], df['sentiment_original'].values[0])])
                print(true_aspects_sentiment, peft_aspect_sentiment, original_aspect_sentiment)
                
                df['true_aspects_sentiment'] = [true_aspects_sentiment]
                df['peft_model_aspect_sentiment'] = [peft_aspect_sentiment]
                df['original_model_aspect_sentiment'] = [original_aspect_sentiment]
                
                true_aspects_sentiment_list.append(true_aspects_sentiment)
                peft_aspect_sentiment_list.append(peft_aspect_sentiment)
                original_aspect_sentiment_list.append(original_aspect_sentiment)
                
                results.append(df.loc[:, ['review', 'true_aspects_sentiment', 'peft_model_aspect_sentiment', 'original_model_aspect_sentiment']])
            
            
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

            st.write("Analysis complete!")
            st.dataframe(pd.concat(results))
            
            st.write('ORIGINAL MODEL: ')
            st.write(original_model_results)
            st.write('PEFT MODEL: ')
            st.write(peft_model_results)
            
            
            
