# pipeline.py
# -*- coding: utf-8 -*-
"""Election Threat Analysis Pipeline with PCA Decomposition"""
import torch
import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel, , GPT2Tokenizer
from hdbscan import HDBSCAN
import annoy
from groq import Groq
from sklearn.decomposition import PCA
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "model_name": "bert-base-multilingual-cased",
    "pca_components": 32,
    "temporal_weight": 0.4,
    "cluster_threshold": 0.35,
    "min_cluster_size": 5,
    "growth_threshold": 50,
    "ann_neighbors": 25,
    "time_window_hours": 48,
    "decay_factor": 0.015,
    "decay_power": 1.8
}

@st.cache_resource(ttl=3600)
def load_model():
    model = BertModel.from_pretrained(CONFIG["model_name"])
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

@st.cache_resource(ttl=3600)
def load_tokenizer():
    return BertTokenizer.from_pretrained(CONFIG["model_name"])

@st.cache_resource(ttl=3600)
def init_groq_client():
    return Groq(api_key="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")

class AnalysisPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer()
        self.model = load_model()
        self.groq_client = init_groq_client()

    @st.cache_data(max_entries=5, ttl=3600, show_spinner="Processing data...")
    def process(_self, df):
        try:
            clean_df = _self.clean_data(df)
            embeddings = _self.generate_embeddings(clean_df['text'].tolist())
            clustered_df = _self.temporal_clustering(clean_df, embeddings)
            analysis_results = _self.analyze_trends(clustered_df)
            analysis_results['viz_figure'] = _self.create_visualization(
                clustered_df, analysis_results['momentum_states'])
            return analysis_results
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    def clean_data(self, df):
        valid_df = df.dropna(subset=["text"])
        valid_df['text'] = valid_df['text'].apply(
            lambda x: re.sub(r"[\x00-\x1F\x7F-\x9F]", "", str(x))
        return valid_df

    @st.cache_data(max_entries=10, ttl=3600)
    def generate_embeddings(_self, texts):
        inputs = _self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(_self.device)

        with torch.no_grad():
            outputs = _self.model(**inputs)
            
        full_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        pca = PCA(n_components=CONFIG["pca_components"])
        return pca.fit_transform(full_embeddings)

    def temporal_clustering(self, df, embeddings):
        timestamps = df['Timestamp'].view('int64').values
        clusters = np.full(len(df), -1, dtype=int)
        ann_index = annoy.AnnoyIndex(embeddings.shape[1], 'euclidean')
        
        for i, emb in enumerate(embeddings):
            ann_index.add_item(i, emb)
        ann_index.build(20)
        
        current_cluster = 0
        chunk_size = 500
        
        for i in range(0, len(embeddings), chunk_size):
            chunk_end = min(i + chunk_size, len(embeddings))
            time_mask = (timestamps >= timestamps[i]) & (timestamps <= timestamps[chunk_end-1])
            
            candidate_indices = []
            for idx in range(i, chunk_end):
                neighbors = ann_index.get_nns_by_item(idx, CONFIG["ann_neighbors"])
                candidate_indices.extend(neighbors)
            
            candidate_indices = np.unique(candidate_indices)
            candidate_indices = candidate_indices[time_mask[candidate_indices]]
            
            if len(candidate_indices) == 0:
                continue
                
            sub_emb = embeddings[candidate_indices]
            sub_ts = timestamps[candidate_indices]
            time_diff = np.abs(sub_ts[:, None] - sub_ts[None, :]) / 3.6e9
            time_mask = (time_diff < CONFIG["time_window_hours"]).astype(float)
            semantic_dists = np.linalg.norm(sub_emb[:, None] - sub_emb[None, :], axis=2)
            combined_dists = (CONFIG["temporal_weight"] * time_diff + 
                            (1 - CONFIG["temporal_weight"]) * semantic_dists) * time_mask
            
            clusterer = HDBSCAN(
                min_cluster_size=CONFIG["min_cluster_size"],
                metric="precomputed",
                cluster_selection_epsilon=CONFIG["cluster_threshold"]
            )
            chunk_clusters = clusterer.fit_predict(combined_dists)
            
            valid_mask = chunk_clusters != -1
            chunk_clusters[valid_mask] += current_cluster
            clusters[candidate_indices] = chunk_clusters
            current_cluster = chunk_clusters[valid_mask].max() + 1 if valid_mask.any() else current_cluster

        df['Cluster'] = clusters
        return df[df['Cluster'] != -1]

    def analyze_trends(self, clustered_df):
        df = clustered_df.copy()
        df['time_window'] = df['Timestamp'].dt.floor("24H")
        emerging = []
        momentum_states = {}
        
        for cluster, cluster_data in df.groupby('Cluster'):
            cluster_data = cluster_data.sort_values('Timestamp')
            momentum = 0
            last_time = None
            sources = set()
            cumulative_activity = 0
            
            for _, row in cluster_data.iterrows():
                if last_time is not None:
                    hours_diff = (row['Timestamp'] - last_time).total_seconds() / 3600
                    decay = np.exp(-CONFIG["decay_factor"] * (hours_diff ** CONFIG["decay_power"]))
                    momentum *= decay
                
                momentum += 1
                sources.add(row['Source'])
                cumulative_activity += 1
                last_time = row['Timestamp']
            
            momentum_score = momentum * len(sources) * np.log1p(cumulative_activity)
            if momentum_score > CONFIG["growth_threshold"] and len(sources) >= 5:
                emerging.append((cluster, momentum_score))
                momentum_states[cluster] = {
                    'momentum': momentum_score,
                    'sources': sources,
                    'last_update': last_time,
                    'cumulative_activity': cumulative_activity,
                    'peak_activity': cluster_data.resample('1H', on='Timestamp').size().max()
                }

        return {
            'emerging_trends': sorted(emerging, key=lambda x: -x[1]),
            'momentum_states': momentum_states
        }

    def create_visualization(self, clustered_df, momentum_states):
        try:
            plt.switch_backend('Agg')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Timeline Plot
            top_clusters = sorted(momentum_states.items(), 
                                key=lambda x: -x[1]['cumulative_activity'])[:5]
            for cluster, metrics in top_clusters:
                cluster_data = clustered_df[clustered_df['Cluster'] == cluster]
                timeline = cluster_data.resample('6H', on='Timestamp').size().cumsum()
                ax1.plot(timeline.index, timeline, label=f"Cluster {cluster}")
            
            ax1.set_title("Narrative Growth Timeline")
            ax1.legend()
            
            # Heatmap
            heatmap_data = pd.pivot_table(
                clustered_df,
                index=pd.Grouper(key='Timestamp', freq='6H'),
                columns='Cluster',
                values='text',
                aggfunc='count',
                fill_value=0
            )
            sns.heatmap(heatmap_data.T, cmap="viridis", ax=ax2)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @st.cache_data(max_entries=20, ttl=3600, show_spinner="Generating report...")
    def generate_investigative_report(_self, cluster_data, momentum_states, cluster_id, max_tokens=1024):
        """Cache generated reports with retry logic"""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        try:
            # Get top 10 documents with URLs
        metrics = momentum_states.get(cluster_id, {})
        sample_docs = cluster_data[['text', 'URL', 'Timestamp']].values.tolist()

        random.shuffle(sample_docs)
        Country="Gabon"


        #report_context = f"""
        #Quantitative Context:
        #- Total Posts: {metrics.get('cumulative_activity', 'N/A')}
        #- Peak Hourly Activity: {metrics.get('peak_activity', 'N/A')}
        #- Unique Sources: {metrics.get('sources', 'N/A')}
        #- Current Momentum Score: {metrics.get('momentum', 'N/A'):.2f}
        #"""
        # Initialize the list of documents to include
        selected_docs = []
        total_tokens = 0

        # Select documents until we hit the token limit
        for doc in sample_docs:
            # Calculate the token count for the document
            doc_tokens = len(tokenizer.encode(doc[0]))  # Encoding only the text

            if total_tokens + doc_tokens <= max_tokens:
                selected_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break
            
            
        response = _self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "system",
                "content": f"""(
                    Generate {Country} structured Foreign/domestic Information Manipulation and Interference (FIMI) intelligence report related to the upcoming presidential elections:

                    -Provide general context and identify key narratives with the reference documents as evidence.\n
                    -Map these narratives lifecycle: First Detected {cluster_data['Timestamp'].min().strftime('%Y-%m-%d %H:%M')} → Last Updated {cluster_data['Timestamp'].max().strftime('%Y-%m-%d %H:%M')}\n
                    -Identify these narratives vehicles like memes, videos or text posts and provide the reference documents\n
                    -Identify primary sources platforms used to spread these narratives\n


                    -Identify and analyse ties and involvement of Russia or China or Turkey or Saudi Arabia.\n
                    -Identify extremism/jihadist cases\n
                    -Identify any case of anti-West, anti-France, pro/anti-ECOWAS, pro/anti-AES (Alliance of Sahel States), pro-Russia or Pro-China sentiment\n
                    -Clearly identify hate speech, negative stereotyping, toxic incitement and mention some of them. Highlight and mention also the corresponding trigger lexicons used\n\n


                    - Identify coordinated network of accounts, analyse network topology and highlight coordination signs like post timing, source distribution, inauthentic engagement spikes on posts. As metrics we have: Total Posts: {metrics.get('cumulative_activity', 'N/A')}, Peak Hourly Activity: {metrics.get('peak_activity', 'N/A')}, source_count: {cluster_data['Source'].nunique()}, Current Momentum Score: {metrics.get('momentum', 'N/A'):.2f}, Timestamp: {cluster_data['Timestamp']}\n
                    - Identify and analyse crossposting clusters\n
                    - Identify narrative engineering like story arc development, meme warfare tactics, sentiment manipulation techniques\n
                    - Identify AI-generated contents mimicking authentic Source\n
                    - Identify reused/manipulated media (e.g., repurposed protest footage from 2021–2024 framed as “current unrest,” AI-generated imagery of alleged government corruption); Identify Viral templates linking policy decisions (e.g., austerity, resource deals) to foreign actors (France/UAE/China/Turkey/Saudi Arabia)\n
                    - Identify Linguistic fingerprints like translation artifacts, atypical local dialect usage\n


                    -Based on all above, suggest 2-3 strong online Investigative leads using using clear, technical and advanced style sentences\n\n
                    Exclude: Speculation, unverified claims, historical background, general statements, findings or answers. Base findings only on provided evidence documents\n
                    Don't include other informations besides what's requested.\n
                    All above insights should be provided relatively to the upcoming presidential elections. Therefore, skip and remove or just add 'non related' on cases or insights or narratives or any patterns that are not related to the upcoming elections.\n
                    Don't duplicate findings from the same documents you are analyzing. Only report NEW patterns not seen in previous analysis.\n
                    Don't use bullet points in the report, only paragraphs: the focus points above are to orient the content of your report not to be used as bullet points.\n
                    Document only what you have found and skip what you didn't find.\n
                    Skip and remove cases or insights or narratives or any patterns that are not related to the upcoming elections.\n
                    Always reference your findings with documents URLs as evidence.\n
                    Reference specific evidence from provided URLs
                )
                """
                }, {
                "role": "user",
                "content": "\n".join([f"Document {i+1}: {doc[0]}\nURL: {doc[1]}" 
                                      for i, doc in enumerate(sample_docs)])
            }],
            temperature=0.6,
            #max_tokens=800,
            timeout=30
        )
        return {
            "report": response.choices[0].message.content,
            "metrics": metrics,
            "sample_texts": [doc[0] for doc in selected_docs],
            "sample_urls": [doc[1] for doc in selected_docs],
            "Time": [doc[2] for doc in selected_docs],
            "all_urls": cluster_data['URL'].head(20).tolist(),
            "source_count": cluster_data['Source'].nunique(),
            "momentum_score": cluster_data['momentum_score'].iloc[0]
            }
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {"error": str(e)}

def categorize_momentum(score):
    score = float(score)
    if score <= 150: return 'Tier 1: Ambient Noise'
    elif score <= 500: return 'Tier 2: Emerging Narrative'
    elif score <= 2000: return 'Tier 3: Coordinated Activity'
    return 'Tier 
