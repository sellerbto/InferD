rm -rf model_parts
uv run split_model.py
uv run generate_docker_compose.py
docker-compose -f docker-compose.generated.yml up --build
# uv run petals/send_message.py