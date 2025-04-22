import csv
import random
import time
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException



API_KEY = ""  
SEARCH_QUERIES = ["music", "technology", "gaming", "travel", "education", "sports", "cooking"]
TARGET_TOTAL_VIDEOS = 500  
COMMENTS_PER_VIDEO = 100
QUOTA_THRESHOLD = 9000  
VIDEOS_PER_CATEGORY = 70  


youtube = build('youtube', 'v3', developerKey=API_KEY)
api_calls_count = 0
DetectorFactory.seed = 0  


CATEGORY_ID_NAME = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
    30: "Movies",
    31: "Anime/Animation",
    32: "Action/Adventure",
    33: "Classics",
    34: "Comedy",
    35: "Documentary",
    36: "Drama",
    37: "Family",
    38: "Foreign",
    39: "Horror",
    40: "Sci-Fi/Fantasy",
    41: "Thriller",
    42: "Shorts",
    43: "Shows",
    44: "Trailers"
}

def get_video_category(video_id):
  
    global api_calls_count
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        api_calls_count += 1
        return response["items"][0]["snippet"]["categoryId"]
    except Exception as e:
        print(f"{video_id} is wrong: {str(e)}")
        return None

def get_video_ids(query, max_videos=70):

    video_ids = []
    next_page_token = None
    global api_calls_count
    
    while len(video_ids) < max_videos:
        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=50,
            pageToken=next_page_token,
            regionCode="US"  
        )
        response = request.execute()
        api_calls_count += 1
        
        for item in response.get("items", []):
            vid = item["id"]["videoId"]
            if vid not in video_ids:
                video_ids.append(vid)
                if len(video_ids) >= max_videos:
                    break
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(1)
    
    return video_ids[:max_videos]  

def get_comments(video_id, category_name):
 
    comments = []
    next_page_token = None
    global api_calls_count
    
    while len(comments) < COMMENTS_PER_VIDEO:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            api_calls_count += 1
        except Exception as e:
            print(f"Fail: {str(e)}")
            break
        
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            try:
                if detect(snippet["textDisplay"]) == "en":
                    comments.append({
                        "video_id": video_id,
                        "user_id": snippet.get("authorChannelId", {}).get("value", ""),
                        "publishedAt": snippet["publishedAt"],
                        "comment_text": snippet["textDisplay"].replace("\n", " "),
                        "likeCount": snippet["likeCount"],
                        "category": category_name
                    })
            except LangDetectException:
                continue
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(1)
    
    return comments

def main():
    all_comments = []
    
    for category_query in SEARCH_QUERIES:
        print(f"\nprocessing : {category_query.upper()}")
        
    
        video_ids = get_video_ids(category_query, VIDEOS_PER_CATEGORY)
        print(f"Find total number of {len(video_ids)}'s video ")
        
    
        for idx, vid in enumerate(video_ids, 1):
            if api_calls_count >= QUOTA_THRESHOLD:
                print("Quote error, stop")
                break
            
            category_id = get_video_category(vid)
            category_name = CATEGORY_ID_NAME.get(int(category_id), "Unknown") if category_id else "Unknown"
            

            
            print(f"processing {idx}/{len(video_ids)}: {vid} ({category_name})")
            comments = get_comments(vid, category_name)
            all_comments.extend(comments)
            time.sleep(1)
        
        if api_calls_count >= QUOTA_THRESHOLD:
            break
    
  
    with open("youtube_comments_with_category.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "user_id", "publishedAt", "comment_text", "likeCount", "category"])
        writer.writeheader()
        writer.writerows(all_comments)
    
    print(f"\nDone! Get the number of {len(all_comments)}'s comments")
  

if __name__ == "__main__":
    main()
