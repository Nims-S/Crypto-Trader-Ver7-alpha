import importlib, os


def test_start_once(monkeypatch):
    monkeypatch.setenv("BOT_THREAD_ENABLED","false")
    import main
    importlib.reload(main)

    calls=[]
    class FakeThread:
        def __init__(self,*a,**k):pass
        def start(self):calls.append(1)

    monkeypatch.setattr(main.threading,"Thread",FakeThread)
    main.BOT_THREAD_STARTED=False

    main.start_background_executor_once()
    main.start_background_executor_once()

    assert len(calls)==1
