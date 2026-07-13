# MNEMOS Collection Setup for E1 Task 01

Use a dedicated MNEMOS collection for the MNEMOS-enabled leg of this task.

Preferred collection:

```text
mnemos_ai_dev_e1_task_01
```

The repository includes a compose override file at:

```text
docker-compose.ai_dev_task_01.override.yml
```

Restart the MNEMOS service on the dedicated collection:

```text
docker compose -f docker-compose.yml -f docker-compose.ai_dev_task_01.override.yml up -d --build mnemos
```

Then seed the task docs:

```text
python tools/seed_mnemos_ai_dev_task_01.py
```

The seeding helper updates `task_control_manifest.json` with:

- active collection name
- task-doc seed snapshot
- collection snapshot
- smoke-check retrieval results

Do not launch the MNEMOS-enabled agent until the seed helper shows the task docs
retrieving correctly.
