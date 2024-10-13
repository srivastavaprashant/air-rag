## AIR RAG Agent
AIR database consists of around 30k records of intracell interaction pairs like protein-protein, etc. This AI agent is an effort to help AIR users to find the relevant information from the database swiftly.


### Using the agent
1. Install python 3.11.*.
2. Install packages using `pip install -r requirements.txt`.
3. Install the biokb package using `poetry install`.
4. Create a '.env' similar to the '.env.example' file.
4. Create a HuggingFace token and add it to the `.env` file.
4. Run the agent using `streamlit run biokb/streamlit/app.py --server.port 8888`.
5. Open the browser and go to `http://localhost:8888`.