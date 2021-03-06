# Поиск похожих разработчиков (учебный проект по Code as data)

### Скрыпников Егор, БПИ192.

Данный проект подразумевает выполнение следующих задач:

1. Сбор данных (списка склонированных репозиториев с Python-, JavaScript- и Ruby-кодом (+ php?)) на github.
2. Подготовка к анализу:
    2.1. Формализация желаемых пространств эмбеддингов для разработчиков. Минимум 2 пространства:
        * Пространство языков программирования
        * Пространство имён переменных и импортов
    2.2. Определить наиболее осмысленные метрики похожести из следующих:
        * Косинусная дистанция
        * L2/L1 меры
        * Containment (как в http://ekzhu.com/datasketch/lshensemble.html#containment).
3. Составление датасета нужного для анализа формата: 
    3.1. Фильтрация репозиториев на предмет именно python-файлов с помощью [enry](https://github.com/go-enry/go-enry).
    3.2. Парсинг этих файлов: вычленение ассоциаций участков кода с разработчиками с помощью [tree-sitter](https://github.com/tree-sitter/tree-sitter)
3. Проведение анализа, в том числе написание анализирующей программы.
4. Представление reproducible анализа в этом гитхаб-репозитории.


How to use:

* Be sure that you are working under a unix-like environment (you'll need `bash` and `git` on your PATH).
* Be sure that you have a python install compatible with all packages in `requirements.txt` and the `enry` package from the `go-enry` github repo.
* Run `build_enry.sh`
* Run `python repoparse.py --repo <either a local repo path or a git-clonable link>`.