from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import logging
import io 

# Load environment variables from .env file
load_dotenv()

# Create app
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for session or flash messages, good practice

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping region codes to country names
region_names = {
    "US": "United States", "PK": "Pakistan", "IN": "India",
    "GB": "United Kingdom", "CA": "Canada", "AU": "Australia",
    "DE": "Germany", "FR": "France", "JP": "Japan", "KR": "South Korea"
}

# Global Constants
API_KEY = os.environ.get('YOUTUBE_API_KEY')
if not API_KEY:
    logging.error("YOUTUBE_API_KEY not found in environment variables.")

PLOT_DIR = 'static/plots'
DATA_DIR = 'data' # Directory to store CSV files

# Ensure the plots and data folders exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def get_trending_videos(api_key, max_results=200, region='US'):
    if not api_key:
        logging.error("API key is missing for get_trending_videos.")
        return []
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        videos = []
        req = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            chart='mostPopular',
            regionCode=region,
            maxResults=50  
        )
        while req and len(videos) < max_results:
            response = req.execute()
            for item in response.get('items', []):
                video_details = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'published_at': item['snippet']['publishedAt'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'category_id': item['snippet']['categoryId'],
                    'tags': item['snippet'].get('tags', []),
                    'duration': item['contentDetails']['duration'],
                    'definition': item['contentDetails']['definition'],
                    'caption': item['contentDetails'].get('caption', 'false'),
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    # 'dislike_count': int(item['statistics'].get('dislikeCount', 0)), # Dislike count is often no longer available
                    'favorite_count': int(item['statistics'].get('favoriteCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0))
                }
                videos.append(video_details)
            req = youtube.videos().list_next(req, response)
        return videos[:max_results]
    except HttpError as e:
        logging.error(f"An API error occurred in get_trending_videos: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_trending_videos: {e}")
        return []

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    logging.info(f"Data saved to {filepath}")
    return filepath 

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logging.warning(f"CSV file {filepath} is empty.")
            return df
        df['description'] = df['description'].fillna('No description')
        df['published_at'] = pd.to_datetime(df['published_at'])
        # Safely evaluate 'tags' column
        def parse_tags(tags_str):
            if isinstance(tags_str, list):
                return tags_str
            if pd.isna(tags_str) or not isinstance(tags_str, str):
                return []
            try:
                return eval(tags_str)
            except (SyntaxError, NameError):
                return [tag.strip() for tag in tags_str.split(',') if tag.strip()] # Fallback for simple comma-separated
        df['tags'] = df['tags'].apply(parse_tags)

        df['duration_seconds'] = df['duration'].apply(lambda x: isodate.parse_duration(x).total_seconds() if pd.notna(x) else 0)
        df['duration_range'] = pd.cut(df['duration_seconds'], bins=[0, 300, 600, 1200, 3600, 7200, float('inf')],
                                      labels=['0-5 min', '5-10 min', '10-20 min', '20-60 min', '60-120 min', '>120 min'],
                                      right=False)
        df['tag_count'] = df['tags'].apply(len)
        df['publish_hour'] = df['published_at'].dt.hour
        return df
    except FileNotFoundError:
        logging.error(f"CSV file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading or preprocessing data from {filepath}: {e}")
        return pd.DataFrame()


def get_category_mapping(api_key, region='US'):
    if not api_key:
        logging.error("API key is missing for get_category_mapping.")
        return {}
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videoCategories().list(part='snippet', regionCode=region)
        response = request.execute()
        return {int(item['id']): item['snippet']['title'] for item in response.get('items', [])}
    except HttpError as e:
        logging.error(f"An API error occurred in get_category_mapping: {e}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_category_mapping: {e}")
        return {}

def map_category_names(df, api_key, region):
    if df.empty or 'category_id' not in df.columns:
        return df
    mapping = get_category_mapping(api_key, region)
    if not mapping: # If mapping is empty due to API error
        df['category_name'] = 'Unknown (API Error)'
        return df
    df['category_name'] = df['category_id'].astype(str).map(lambda x: mapping.get(int(x), 'Unknown'))
    return df


def plot_distributions(df):
    if df.empty or df[['view_count', 'like_count', 'comment_count']].isnull().all().all():
        logging.warning("Skipping plot_distributions due to empty or all-NaN data.")
        return
    # View Count Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['view_count'].dropna(), bins=30, kde=True, color='blue')
    plt.title('View Count Distribution')
    plt.xlabel('View Count')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PLOT_DIR, 'view_count_distribution.png'))
    plt.close()

    # Like Count Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['like_count'].dropna(), bins=30, kde=True, color='green')
    plt.title('Like Count Distribution')
    plt.xlabel('Like Count')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PLOT_DIR, 'like_count_distribution.png'))
    plt.close()

    # Comment Count Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['comment_count'].dropna(), bins=30, kde=True, color='red')
    plt.title('Comment Count Distribution')
    plt.xlabel('Comment Count')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PLOT_DIR, 'comment_count_distribution.png'))
    plt.close()
    logging.info("Distribution plots generated.")


def plot_correlation_matrix(df):
    if df.empty or len(df[['view_count', 'like_count', 'comment_count']].dropna()) < 2:
        logging.warning("Skipping correlation matrix due to insufficient data.")
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['view_count', 'like_count', 'comment_count']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Engagement Metrics')
    plt.savefig(os.path.join(PLOT_DIR, 'correlation_matrix.png'))
    plt.close()
    logging.info("Correlation matrix plot generated.")

def plot_category_distribution(df):
    if df.empty or 'category_name' not in df.columns or df['category_name'].isnull().all():
        logging.warning("Skipping category distribution plot due to missing category data.")
        return
    plt.figure(figsize=(12, 8))
    sns.countplot(y=df['category_name'], order=df['category_name'].value_counts().index, palette='viridis')
    plt.title('Number of Trending Videos by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'category_distribution.png'))
    plt.close()
    logging.info("Category distribution plot generated.")

def plot_engagement_by_category(df):
    if df.empty or 'category_name' not in df.columns or df['category_name'].isnull().all():
        logging.warning("Skipping engagement by category plot due to missing category data.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    metrics = ['view_count', 'like_count', 'comment_count']
    titles = ['Average View Count', 'Average Like Count', 'Average Comment Count']
    
    # Drop rows where all metrics are NaN to prevent errors in groupby().mean()
    df_metrics = df[['category_name'] + metrics].dropna(subset=metrics, how='all')
    if df_metrics.empty:
        logging.warning("No valid data for engagement by category plot after dropping NaNs.")
        plt.close(fig) 
        return

    category_engagement = df_metrics.groupby('category_name')[metrics].mean().sort_values(by='view_count', ascending=False)
    
    if category_engagement.empty:
        logging.warning("Category engagement data is empty after groupby.")
        plt.close(fig)
        return

    for i, metric in enumerate(metrics):
        sns.barplot(y=category_engagement.index, x=category_engagement[metric], ax=axes[i], palette='viridis', errorbar=None) 
        axes[i].set_title(titles[i] + ' by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'engagement_by_category.png'))
    plt.close()
    logging.info("Engagement by category plot generated.")


def plot_video_length_vs_views(df):
    if df.empty or 'duration_seconds' not in df.columns or 'view_count' not in df.columns or \
       df[['duration_seconds', 'view_count']].isnull().all().all():
        logging.warning("Skipping video length vs views plot due to missing data.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration_seconds', y='view_count', data=df.dropna(subset=['duration_seconds', 'view_count']), alpha=0.6, color='purple')
    plt.title('Video Length vs View Count')
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('View Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'video_length_vs_views.png'))
    plt.close()
    logging.info("Video length vs views plot generated.")

def plot_engagement_by_duration(df):
    if df.empty or 'duration_range' not in df.columns or df['duration_range'].isnull().all():
        logging.warning("Skipping engagement by duration plot due to missing duration range data.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    metrics = ['view_count', 'like_count', 'comment_count']
    titles = ['Average View Count', 'Average Like Count', 'Average Comment Count']
    
    duration_order = ['0-5 min', '5-10 min', '10-20 min', '20-60 min', '60-120 min', '>120 min']
    
    # Drop rows where all metrics are NaN
    df_metrics = df[['duration_range'] + metrics].dropna(subset=metrics, how='all')
    if df_metrics.empty:
        logging.warning("No valid data for engagement by duration plot after dropping NaNs.")
        plt.close(fig)
        return

    length_engagement = df_metrics.groupby('duration_range', observed=False)[metrics].mean().reindex(duration_order)

    if length_engagement.empty:
        logging.warning("Length engagement data is empty after groupby/reindex.")
        plt.close(fig)
        return

    for i, metric in enumerate(metrics):
        sns.barplot(y=length_engagement.index, x=length_engagement[metric], ax=axes[i], palette='magma', errorbar=None) 
        axes[i].set_title(titles[i] + ' by Duration Range')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'engagement_by_duration.png'))
    plt.close()
    logging.info("Engagement by duration plot generated.")

def plot_tag_count_vs_views(df):
    if df.empty or 'tag_count' not in df.columns or 'view_count' not in df.columns or \
       df[['tag_count', 'view_count']].isnull().all().all():
        logging.warning("Skipping tag count vs views plot due to missing data.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tag_count', y='view_count', data=df.dropna(subset=['tag_count', 'view_count']), alpha=0.6, color='orange')
    plt.title('Number of Tags vs View Count')
    plt.xlabel('Number of Tags')
    plt.ylabel('View Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tag_count_vs_views.png'))
    plt.close()
    logging.info("Tag count vs views plot generated.")

def plot_publish_hour_distribution(df):
    if df.empty or 'publish_hour' not in df.columns or df['publish_hour'].isnull().all():
        logging.warning("Skipping publish hour distribution plot due to missing data.")
        return
    plt.figure(figsize=(12, 6))
    sns.countplot(x='publish_hour', data=df.dropna(subset=['publish_hour']), palette='coolwarm', order = sorted(df['publish_hour'].dropna().unique().astype(int)))
    plt.title('Distribution of Videos by Publish Hour')
    plt.xlabel('Publish Hour (24-hour format)')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'publish_hour_distribution.png'))
    plt.close()
    logging.info("Publish hour distribution plot generated.")

def plot_publish_hour_vs_views(df):
    if df.empty or 'publish_hour' not in df.columns or 'view_count' not in df.columns or \
       df[['publish_hour', 'view_count']].isnull().all().all():
        logging.warning("Skipping publish hour vs views plot due to missing data.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='publish_hour', y='view_count', data=df.dropna(subset=['publish_hour', 'view_count']), alpha=0.6, color='teal')
    plt.title('Publish Hour vs View Count')
    plt.xlabel('Publish Hour (24-hour format)')
    plt.ylabel('View Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'publish_hour_vs_views.png'))
    plt.close()
    logging.info("Publish hour vs views plot generated.")


def generate_conclusions(df, country_name):
    conclusions = []
    if df.empty:
        conclusions.append(f"We couldn't generate detailed insights for {country_name} because no data was available for analysis.")
        return conclusions

    total_videos = len(df)
    
    # --- General Overview ---
    if total_videos > 0:
        avg_views = df['view_count'].mean()
        avg_likes = df['like_count'].mean()
        avg_comments = df['comment_count'].mean()
        conclusions.append(
            f"Overall Trends: For the {total_videos} trending videos in {country_name}, videos generally receive around "
            f"{avg_views:,.0f} views, {avg_likes:,.0f} likes, and {avg_comments:,.0f} comments."
        )

    # --- Engagement Correlation ---
    if not df[['view_count', 'like_count']].isnull().all().all() and len(df[['view_count', 'like_count']].dropna()) >= 2:
        correlation = df['view_count'].corr(df['like_count'])
        if pd.notna(correlation):
            if correlation > 0.7:
                corr_desc = "a strong connection"
            elif correlation > 0.3:
                corr_desc = "a moderate connection"
            else:
                corr_desc = "a weak connection"
            conclusions.append(
                f"Engagement Insight: There's generally {corr_desc} between how many views a video gets and how many likes it receives "
                f"({correlation:.2f} correlation). This means videos with more views often also have more likes, and vice-versa."
            )

    # --- Top Categories ---
    if 'category_name' in df.columns and not df['category_name'].empty and not df['category_name'].isnull().all():
        top_categories = df['category_name'].value_counts().head(3)
        if not top_categories.empty:
            category_list = ', '.join([f"'{cat}'" for cat in top_categories.index])
            conclusions.append(
                f"Most Popular Content: The top trending video categories in {country_name} are typically: {category_list}. "
                "This indicates what type of content is currently resonating most with viewers."
            )
            
            category_engagement = df.groupby('category_name')[['view_count', 'like_count']].mean()
            if not category_engagement.empty:
                top_viewed_category = category_engagement['view_count'].idxmax() if not category_engagement['view_count'].empty else None
                top_liked_category = category_engagement['like_count'].idxmax() if not category_engagement['like_count'].empty else None
                
                if top_viewed_category and top_liked_category and top_viewed_category == top_liked_category:
                    conclusions.append(f"Category Performance: The category '{top_viewed_category}' stands out, not only for having many trending videos but also for generating the highest average views and likes.")
                elif top_viewed_category and top_liked_category:
                    conclusions.append(f"Category Performance: While '{top_viewed_category}' generally attracts the most views, '{top_liked_category}' often receives the most likes on average. This might suggest different engagement patterns across categories.")


    # --- Video Duration Impact ---
    if 'duration_range' in df.columns and not df['duration_range'].empty and not df['duration_range'].isnull().all():
        engagement_by_duration = df.groupby('duration_range', observed=False)['view_count'].mean()
        if not engagement_by_duration.empty and engagement_by_duration.max() > 0:
            optimal_duration = engagement_by_duration.idxmax()
            max_views_duration = engagement_by_duration.max()
            conclusions.append(
                f"Video Length Sweet Spot: Videos falling within the '{optimal_duration}' length tend to get the highest average view counts "
                f"in {country_name}. This suggests there might be an optimal video length for maximizing reach."
            )
            # Add a secondary observation if applicable
            sorted_durations = engagement_by_duration.sort_values(ascending=False)
            if len(sorted_durations) > 1 and sorted_durations.iloc[1] > 0:
                second_best_duration = sorted_durations.index[1]
                if sorted_durations.iloc[0] / sorted_durations.iloc[1] > 1.5: 
                    conclusions.append(f"Consider focusing on videos around '{optimal_duration}' as they show significantly higher engagement.")
                else:
                    conclusions.append(f"Both '{optimal_duration}' and '{second_best_duration}' duration ranges show strong viewer engagement.")


    # --- Publishing Time Analysis ---
    if 'publish_hour' in df.columns and not df['publish_hour'].empty and not df['publish_hour'].isnull().all():
        publish_hour_counts = df['publish_hour'].value_counts().sort_index()
        if not publish_hour_counts.empty:
            peak_hours = publish_hour_counts.nlargest(2).index.tolist()
            
            if peak_hours:
                peak_hours_str = f"{int(peak_hours[0])}:00"
                if len(peak_hours) > 1:
                    peak_hours_str += f" and {int(peak_hours[1])}:00"
                
                conclusions.append(
                    f"Best Time to Publish: Trending videos in {country_name} are most often published around "
                    f"{peak_hours_str} (UTC). This could indicate popular content release times or audience availability."
                )
                
                avg_views_by_hour = df.groupby('publish_hour')['view_count'].mean()
                if not avg_views_by_hour.empty and avg_views_by_hour.max() > 0:
                    best_hour_for_views = int(avg_views_by_hour.idxmax())
                    conclusions.append(
                        f"Videos published around {best_hour_for_views}:00 (UTC) tend to gather the highest average view counts, suggesting this might be a prime time to reach viewers."
                    )


    # --- Tag Usage Effectiveness ---
    if 'tag_count' in df.columns and not df['tag_count'].empty and not df['tag_count'].isnull().all():
        avg_tags = df['tag_count'].mean()
        conclusion_tag = f"Tagging Strategy: On average, trending videos use about {avg_tags:.1f} tags. "
        if not df[['tag_count', 'view_count']].isnull().all().all() and len(df[['tag_count', 'view_count']].dropna()) >= 2:
            views_vs_tags_corr = df['tag_count'].corr(df['view_count'])
            if pd.notna(views_vs_tags_corr):
                if views_vs_tags_corr > 0.2:
                    conclusion_tag += "There's a noticeable positive correlation, meaning more tags can lead to more views."
                elif views_vs_tags_corr < -0.2:
                    conclusion_tag += "Interestingly, there's a slight negative correlation, suggesting too many tags might not always be beneficial."
                else:
                    conclusion_tag += "The number of tags seems to have a weak direct relationship with view counts."
        conclusions.append(conclusion_tag)

    if not conclusions:
        conclusions.append(f"We couldn't find enough clear patterns or insights from the data for {country_name} at this time. More data might be needed for a comprehensive analysis.")


    return conclusions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        region_code = request.form.get('region')
        if not region_code:
            return render_template('index.html', error_message="Please select a region.")

        country_name = region_names.get(region_code, "Selected Region")
        logging.info(f"Analysis requested for region: {region_code} ({country_name})")

        csv_filename = f'trending_videos_{region_code}.csv'
        csv_filepath = os.path.join(DATA_DIR, csv_filename)


        trending_videos = get_trending_videos(API_KEY, region=region_code, max_results=200)
        if not trending_videos:
            logging.error(f"No trending videos data fetched for region {region_code}.")
            return render_template('index.html', error_message=f"Could not fetch trending video data for {country_name}. API limit might be reached or no data available.")
        
        save_to_csv(trending_videos, csv_filename) 

        df = load_and_preprocess_data(csv_filepath)
        if df.empty:
            logging.error(f"DataFrame is empty after loading/preprocessing for {region_code}.")
            return render_template('index.html', error_message=f"Failed to process data for {country_name}. The data file might be empty or corrupted.")

        df = map_category_names(df, API_KEY, region_code)

        # Generate plots
        plot_distributions(df)
        plot_correlation_matrix(df)
        plot_category_distribution(df)
        plot_engagement_by_category(df)
        plot_video_length_vs_views(df)
        plot_engagement_by_duration(df)
        plot_tag_count_vs_views(df)
        plot_publish_hour_distribution(df)
        plot_publish_hour_vs_views(df)
        
        analysis_conclusions = generate_conclusions(df, country_name)

        return render_template('results.html', 
                               country=country_name, 
                               conclusions=analysis_conclusions,
                               csv_filename=csv_filename, # Pass filename for download link
                               region_code=region_code) 

    return render_template('index.html', error_message=None)


@app.route('/download/<region_code>')
def download_csv(region_code):
    filename = f'trending_videos_{region_code}.csv'
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    else:
        logging.error(f"Download requested for non-existent file: {filepath}")
        return redirect(url_for('index', error_message="CSV file not found for download."))


if __name__ == "__main__":
    if not API_KEY:
        print("CRITICAL ERROR: YOUTUBE_API_KEY is not set. Please set it in your .env file.")
    else:
        print("YOUTUBE_API_KEY loaded successfully.")
    app.run(debug=True)