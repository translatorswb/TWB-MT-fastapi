from app import create_app

app = create_app()
celery = app.celery_app

def celery_worker():
    from watchgod import run_process
    import subprocess

    def run_worker():
        subprocess.call(
            ["celery", "-A", "main.celery", "worker", "--loglevel=info", "--max-tasks-per-child=1", "--autoscale=1,2"]
        )

    run_process("./app", run_worker)


if __name__ == "__main__":
    celery_worker()
