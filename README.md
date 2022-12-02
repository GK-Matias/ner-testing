# ner-testing
By following these steps I get different training results (printed to stdout) on different computers

```bash
docker build . -t ner-testing
```

```bash
docker run ner-testing
```

```bash
docker exec -it <container id> /bin/bash
```

```bash
python train_test_ner_model.py
```


The `test_ner_model.py` file is intended for testing two models that are saved to disk.