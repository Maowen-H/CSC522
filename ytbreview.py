# import csv
# import random
# import time
# from googleapiclient.discovery import build

# # -------------------------------
# # Setting API
# # -------------------------------
# API_KEY = "AIzaSyDWbCLeL8fTdLAQrGsTHp71esEYECG8lVY"  
# SEARCH_QUERIES = ["music", "technology", "gaming", "travel", "education", "sports", "cooking"]
# SEARCH_QUERY = random.choice(SEARCH_QUERIES)
# TARGET_VIDEO_COUNT = 500  
# COMMENTS_PER_VIDEO = 100  

# QUOTA_THRESHOLD = 3000


# youtube = build('youtube', 'v3', developerKey=API_KEY)


# api_calls_count = 0


# def get_video_ids(query, max_videos=500):
#     """
#     使用 search.list 接口根据 query 获取视频 ID 列表。
#     每次请求返回最多 50 个结果，循环直到达到目标视频数量或无更多结果。
#     视频 ID 去重。
#     """
#     video_ids = []
#     next_page_token = None
#     global api_calls_count

#     while len(video_ids) < max_videos:
#         request = youtube.search().list(
#             part="id",
#             q=query,
#             type="video",
#             maxResults=50,
#             pageToken=next_page_token
#         )
#         response = request.execute()
#         api_calls_count += 1  # 记录一次 API 调用

#         for item in response.get("items", []):
#             vid = item["id"]["videoId"]
#             if vid not in video_ids:  # 保证去重
#                 video_ids.append(vid)
#                 if len(video_ids) >= max_videos:
#                     break

#         next_page_token = response.get("nextPageToken")
#         if not next_page_token:
#             break  # 没有更多结果则退出循环

#         time.sleep(5)  # 稍作延时，避免请求过快

#     return video_ids

# def get_comments(video_id, max_comments=100):
#     """
#     使用 commentThreads.list 接口获取单个视频的评论。
#     如果评论不足 max_comments，则获取所有评论。
#     每条评论提取：用户 ID、发布时间、评论内容、点赞数。
#     """
#     comments_list = []
#     next_page_token = None
#     global api_calls_count

#     while len(comments_list) < max_comments:
#         try:
#             request = youtube.commentThreads().list(
#                 part="snippet",
#                 videoId=video_id,
#                 maxResults=100,  # 接口允许一次返回最多 100 条评论
#                 pageToken=next_page_token,
#                 textFormat="plainText"
#             )
#             response = request.execute()
#             api_calls_count += 1  # 记录一次 API 调用
#         except Exception as e:
#             print(f"Get {video_id} review error :{e}")
#             break

#         items = response.get("items", [])
#         if not items:
#             # 视频没有评论
#             break

#         for item in items:
#             snippet = item["snippet"]["topLevelComment"]["snippet"]
#             comment_data = {
#                 "video_id": video_id,
#                 "user_id": snippet.get("authorChannelId", {}).get("value", ""),
#                 "publishedAt": snippet.get("publishedAt", ""),
#                 "comment_text": snippet.get("textDisplay", ""),
#                 "likeCount": snippet.get("likeCount", 0)
#             }
#             comments_list.append(comment_data)
#             if len(comments_list) >= max_comments:
#                 break

#         next_page_token = response.get("nextPageToken")
#         if not next_page_token:
#             break  # 无更多评论则退出

#         time.sleep(5)  # 避免过快请求

#     return comments_list

# # -------------------------------
# # 主流程
# # -------------------------------
# def main():
#     global api_calls_count
#     print("Searching for videos based on keywords, collecting video IDs...")
#     video_ids = get_video_ids(SEARCH_QUERY, max_videos=TARGET_VIDEO_COUNT)
#     print(f"Find the total number of {len(video_ids)} videos")

#     # 为了随机性可以打乱顺序
#     random.shuffle(video_ids)

#     all_comments = []
#     for idx, vid in enumerate(video_ids, start=1):
#         # 检查 API 调用数是否接近阈值
#         if api_calls_count >= QUOTA_THRESHOLD:
#             print("The number of API calls has reached the threshold, stopping further requests.")
#             break

#         print(f"({idx}/{len(video_ids)}) Processing video {vid} ...")
#         comments = get_comments(vid, max_comments=COMMENTS_PER_VIDEO)
#         if len(comments) < COMMENTS_PER_VIDEO:
#             print(f"Video {vid} only has {len(comments)} reviews")
#         else:
#             print(f"Video {vid} successfully get {len(comments)} reviews")
#         all_comments.extend(comments)

#         time.sleep(3)  # 避免连续请求过快

#     # 保存到 CSV 文件
#     output_file = "youtube_comments.csv"
#     with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
#         fieldnames = ["video_id", "user_id", "publishedAt", "comment_text", "likeCount"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(all_comments)

#     print(f"The crawl is complete, with a total number of API calls:{api_calls_count}")
#     print(f"A total of {len(all_comments)} comments were collected and the results were saved to {output_file}")

# if __name__ == "__main__":
#     main()
import csv
import random
import time
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# -------------------------------
# 设置 API
# -------------------------------
API_KEY = ""
SEARCH_QUERIES = ["music", "technology", "gaming", "travel", "education", "sports", "cooking"]
SEARCH_QUERY = random.choice(SEARCH_QUERIES)
TARGET_VIDEO_COUNT = 500  
COMMENTS_PER_VIDEO = 100  

QUOTA_THRESHOLD = 3000


youtube = build('youtube', 'v3', developerKey=API_KEY)

api_calls_count = 0

# 用于确保 langdetect 的一致性
DetectorFactory.seed = 0

def get_video_ids(query, max_videos=500):
    """
    使用 search.list 接口根据 query 获取视频 ID 列表。
    每次请求返回最多 50 个结果，循环直到达到目标视频数量或无更多结果。
    视频 ID 去重。
    """
    video_ids = []
    next_page_token = None
    global api_calls_count

    while len(video_ids) < max_videos:
        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        api_calls_count += 1  # 记录一次 API 调用

        for item in response.get("items", []):
            vid = item["id"]["videoId"]
            if vid not in video_ids:  # 保证去重
                video_ids.append(vid)
                if len(video_ids) >= max_videos:
                    break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # 没有更多结果则退出循环

        time.sleep(5)  # 稍作延时，避免请求过快

    return video_ids

def get_comments(video_id, max_comments=100):
    """
    使用 commentThreads.list 接口获取单个视频的评论。
    如果评论不足 max_comments，则获取所有评论。
    每条评论提取：用户 ID、发布时间、评论内容、点赞数。
    """
    comments_list = []
    next_page_token = None
    global api_calls_count

    while len(comments_list) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # 接口允许一次返回最多 100 条评论
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            api_calls_count += 1  # 记录一次 API 调用
        except Exception as e:
            print(f"Get {video_id} review error :{e}")
            break

        items = response.get("items", [])
        if not items:
            # 视频没有评论
            break

        for item in items:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_data = {
                "video_id": video_id,
                "user_id": snippet.get("authorChannelId", {}).get("value", ""),
                "publishedAt": snippet.get("publishedAt", ""),
                "comment_text": snippet.get("textDisplay", ""),
                "likeCount": snippet.get("likeCount", 0)
            }
            # 使用 langdetect 判断评论是否为英文
            try:
                lang = detect(comment_data["comment_text"])
                if lang == "en":  # 只处理英文评论
                    comments_list.append(comment_data)
            except LangDetectException:
                continue  # 如果无法识别语言，跳过该评论

            if len(comments_list) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # 无更多评论则退出

        time.sleep(5)  # 避免过快请求

    return comments_list

# -------------------------------
# 主流程
# -------------------------------
def main():
    global api_calls_count
    print("Searching for videos based on keywords, collecting video IDs...")
    video_ids = get_video_ids(SEARCH_QUERY, max_videos=TARGET_VIDEO_COUNT)
    print(f"Find the total number of {len(video_ids)} videos")

    # 为了随机性可以打乱顺序
    random.shuffle(video_ids)

    all_comments = []
    for idx, vid in enumerate(video_ids, start=1):
        # 检查 API 调用数是否接近阈值
        if api_calls_count >= QUOTA_THRESHOLD:
            print("The number of API calls has reached the threshold, stopping further requests.")
            break

        print(f"({idx}/{len(video_ids)}) Processing video {vid} ...")
        comments = get_comments(vid, max_comments=COMMENTS_PER_VIDEO)
        if len(comments) < COMMENTS_PER_VIDEO:
            print(f"Video {vid} only has {len(comments)} reviews")
        else:
            print(f"Video {vid} successfully get {len(comments)} reviews")
        all_comments.extend(comments)

        time.sleep(3)  # 避免连续请求过快

    # 保存到 CSV 文件
    output_file = "youtube_comments2.csv"
    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["video_id", "user_id", "publishedAt", "comment_text", "likeCount"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comments)

    print(f"The crawl is complete, with a total number of API calls:{api_calls_count}")
    print(f"A total of {len(all_comments)} comments were collected and the results were saved to {output_file}")

if __name__ == "__main__":
    main()
