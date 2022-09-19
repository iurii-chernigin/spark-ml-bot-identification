# Bot identificaton based on statistics of user sessions
## Tools
- Spark ML
## Context
We have statistics on user sessions, where some of them are [web crawlers](https://en.wikipedia.org/wiki/Web_crawler), we need to identification such sessions to prevent bot activity.
### Data

| Field  | Description |
| ------------- | ------------- |
| session_id | ID of user session  |
| user_type | User type: authorized or guest |
| duration | Session duration in secs |
| platform | User platform: web, ios, android |
| item_info_events | Number of product information views per session |
| select_item_events | Number of product selects per session |
| make_order_events | Number of product orders per session |
| events_per_min | Average number of events per minutes for session |
| is_bot | 0 is user, 1 is bot |
