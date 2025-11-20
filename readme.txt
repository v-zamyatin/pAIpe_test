•  Директория docker/ — содержит Dockerfile с шагами сборки Docker-образа для развёртывания ПО.
    Dockerfile 
•  Директория data/ — входные и вспомогательные данные:
    data/CADD/whole_genome_SNVs.tsv.gz — оценки CADD (GRCh38).
    data/CADD/whole_genome_SNVs.tsv.gz.tbi — индекс для файла CADD.
    data/GPN-MSA/scores.tsv.bgz — оценки GPN-MSA (HG38).
    data/GPN-MSA/scores.tsv.bgz.tbi — индекс для файла GPN-MSA.
    data/TraitGym/TG.txt — подготовленный датасет TraitGym для обучения.
    data/tracks_ontology_IDs_RNA-seq_with_groups.xlsx — словарь/категории тканей для постобработки AlphaGenome.
•  Директория preprocessed_data/ — промежуточные результаты препроцессинга и признаков:
    preprocessed_data/train/
        borzoi_input.vcf — VCF для запуска Borzoi на обучающих данных.
        sad.h5 — выход Borzoi (лог-метрики по трекам) для обучения.
        borzoi_tracks.txt — табличный вывод признаков Borzoi (train).
        cadd_subset.txt — срез CADD по позициям обучающих вариантов.
        gpn_msa_subset.txt — срез GPN-MSA по позициям обучающих вариантов.
        borzoi_with_cadd_gpn_msa.txt — объединённые признаки (Borzoi + CADD + GPN-MSA) для обучения.
        borzoi_with_cadd_gpn_msa_clf.txt — то же + столбец с меткой класса (train, для обучения классификатора).
    preprocessed_data/test/
        borzoi_input.vcf — VCF для запуска Borzoi на данных пользователя.
        sad.h5 — выход Borzoi (лог-метрики по трекам) для вариантов пользователя.
        borzoi_tracks.txt — табличный вывод признаков Borzoi (test).
        cadd_subset.txt — срез CADD по позициям пользовательских вариантов.
        gpn_msa_subset.txt — срез GPN-MSA по позициям пользовательских вариантов.
        borzoi_with_cadd_gpn_msa.txt — объединённые признаки (Borzoi + CADD + GPN-MSA) для применения модели.
        sed.h5 — выход Borzoi SED (ген-уровень, влияние на экспрессию) для вариантов пользователя.
        borzoi_genes.txt — агрегированные максимальные изменения экспрессии по кардио-тканям (для пользователя).
        borzoi_score.txt — сводная таблица: максимальное влияние по тканям + вероятности патогенности (для пользователя).
•  Директория results/ — результаты обучения и инференса:
    final_model.joblib — обученная модель.
    prauc_per_chrom.csv — AUPRC по каждой хромосоме (LOCO-валидация).
    final_model_mean_coefficients.csv — усреднённые веса финальной модели.
    f1_threshold_curve.png — кривая F1 в зависимости от порога.
    probs_distribution.png — распределение вероятностей по классам.
    roc_curve.png — ROC-кривая.
    pr_curve.png — PR-кривая.
    predictions_new.txt — вероятности патогенности для пользовательских вариантов.
    alphagenome_raw_scores.csv — сырые результаты AlphaGenome.
    alphagenome_score.txt — агрегированные оценки AlphaGenome по выбранным тканям.
    borz_score_ag.txt — финальная сводная таблица (Borzoi + AlphaGenome + патогенность).
•  Корневые файлы:
    input_variants.txt — входные варианты пользователя (CHROM, POS, REF, ALT).
    01_create_docker_image.sh — Установка Docker и настройка окружения.
    02_prepare_and_download.py — Предподготовка и загрузка данных.
    03_train_model.py — Обучение модели.
    04_apply_models.py — Запуск моделей Borzoi, AlphaGenome и ансамблевой модели ИИ.
    readme.txt - файловая структура программного комплекса
    run_docker.sh - запуск docker-контейнера
