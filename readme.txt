(1) Open terminal and Run python ingest.py 
(2) Run python chat_api.py
Description: It will start server process.
(3) Open new terminal and run "Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method Post -ContentType "application/json" -Body '{"session_id": "session_001", "message": "What is your return policy?"}'"
Description : It will give you session_id, intent response and sources. You can change question according to your demand.
(4) To shut down application type Ctrl + C on terminal 1, it will stop application 