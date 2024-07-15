from datasets import load_dataset
import random
from ranking_challenge.fake import fake_request

comments = fake_request(n_posts=1, n_comments=2)

# comments = {
#     "session": {
#         "session_id": "719f30a1-03bb-4d41-a654-138da5c43547",
#         "user_id": "193a9e01-8849-4e1f-a42a-a859fa7f2ad3",
#         "user_name_hash": "6511c5688bbb87798128695a283411a26da532df06e6e931a53416e379ddda0e",
#         "platform": "reddit",
#         "cohort": "AB",
#         "cohort_index": 1,
#         "url": "https://reddit.com/r/PRCExample/1f4deg/example_to_insert",
#         "current_time": "2024-01-20 18:41:20",
#     },
#     "items": [
#         {
#             "id": "de83fc78-d648-444e-b20d-853bf05e4f0e",
#             "title": "this is the post title, available only on reddit",
#             "text": "this is the worst thing I have ever seen!",
#             "author_name_hash": "60b46b7370f80735a06b7aa8c4eb6bd588440816b086d5ef7355cf202a118305",
#             "embedded_urls": [],
#             "type": "post",
#             "created_at": "2023-12-06 17:02:11",
#             "engagements": {"upvote": 34, "downvote": 27, "comment": 20, "award": 0},
#         },
#         {
#             "id": "s5ad13266-8abk4-5219-kre5-2811022l7e43dv",
#             "post_id": "de83fc78-d648-444e-b20d-853bf05e4f0e",
#             "parent_id": "",
#             "text": "this is amazing!",
#             "author_name_hash": "60b46b7370f80735a06b7aa8c4eb6bd588440816b086d5ef7355cf202a118305",
#             "embedded_urls": [],
#             "type": "comment",
#             "created_at": "2023-12-08 11:32:12",
#             "engagements": {"upvote": 15, "downvote": 2, "comment": 22, "award": 2},
#         },
#         {
#             "id": "a4c08177-8db2-4507-acc1-1298220be98d",
#             "post_id": "de83fc78-d648-444e-b20d-853bf05e4f0e",
#             "parent_id": "s5ad13266-8abk4-5219-kre5-2811022l7e43dv",
#             "text": "this thing is ok.",
#             "author_name_hash": "60b46b7370f80735a06b7aa8c4eb6bd588440816b086d5ef7355cf202a118305",
#             "embedded_urls": [],
#             "type": "comment",
#             "created_at": "2023-12-08 11:35:00",
#             "engagements": {"upvote": 3, "downvote": 5, "comment": 10, "award": 0},
#         },
#     ],
# }