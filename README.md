# big-data-click-through-rate-analysis
### Week 1 Задачи Data Analysis ClickHouse

1. Получить статистику по дням. Просто посчитать число всех событий по дням, число показов, число кликов, число уникальных объявлений и уникальных кампаний.
2. Разобраться, почему случился такой скачок 2019-04-05? Каких событий стало больше? У всех объявлений или только у некоторых?
3. Найти топ 10 объявлений по CTR за все время. CTR — это отношение всех кликов объявлений к просмотрам. Например, если у объявления было 100 показов и 2 клика, CTR = 0.02. Различается ли средний и медианный CTR объявлений в наших данных?
4. Похоже, в наших логах есть баг, объявления приходят с кликами, но без показов! Сколько таких объявлений, есть ли какие-то закономерности? Эта проблема наблюдается на всех платформах?
5. Есть ли различия в CTR у объявлений с видео и без? А чему равняется 95 процентиль CTR по всем объявлениям за 2019-04-04?
6. Для финансового отчета нужно рассчитать наш заработок по дням. В какой день мы заработали больше всего? В какой меньше? Мы списываем с клиентов деньги, если произошел клик по CPC объявлению, и мы списываем деньги за каждый показ CPM объявления, если у CPM объявления цена - 200 рублей, то за один показ мы зарабатываем 200 / 1000.
7. Какая платформа самая популярная для размещения рекламных объявлений? Сколько процентов показов приходится на каждую из платформ (колонка platform)?


### Week 2 Задачи Data Engineering

Необходимо реализовать на PySpark задачу обработки данных для их подготовки к обучению моделей.

В результате выполнения задачи, например, выполнив команду:

`spark-submit PySparkJob.py clickstream.parquet result`
или 

`python PySparkJob.py clickstream.parquet result`
Вы должны прочитать указанный в параметрах файл, обработать его и получить структуру папок вида:

/result/train

/result/test

/result/validate

С наборами данных в следующем соотношении train/test/validate = 0.5/0.25/0.25 (randomSplit).
Где в каждой папке должен находиться parquet-файл (число партиций не принципиально) со следующей структурой данных:

| Имя колонки	| Тип колонки	| Описание |
| ----------- | ----------- | -------- |
|ad_id | integer |	id рекламного объявления|
| target_audience_count	| decimal	| размер аудитории, на которую таргетируется объявление |
|has_video	| integer |	1 если есть видео, иначе 0|
|is_cpm	| integer	| 1 если тип объявления CPM, иначе 0|
|is_cpc |	integer	| 1 если тип объявления CPC, иначе 0|
|ad_cost	| double | стоимость объявления в рублях|
|day_count	| integer	| Число дней, которое показывалась реклама|
|CTR	| double |	Отношение числа кликов к числу просмотров|

### Week 3 Задачи ML Engineering

Ваша команда ученых по данным работала с маленькой выборкой данных в 4Гб от общего датасета данных и рекомендует применять линейную регрессию со следующими параметрами:

maxIter=40, regParam=0.4, elasticNetParam=0.8

Однако вам необходимо реализовать распределенную модель, выбрать тип из возможных, для этого, вы можете задействовать любую модель регрессии (DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, подобрать оптимальные гиперпараметры и сравнить результаты.
Как результат реализуйте две PySpark задачи и загрузите их в ответ(можно добавить Jupyter Notebook с анализом моделей):

1) PySparkMLFit.py - задача, которая должна тренировать модель, подбирать оптимальные гиперпараметры на входящих данных, сохранять ее и производить оценку качества модели, используя RegressionEvaluator и выводя в консоль RMSE модели на основе test датасета.
Варианты запуска задачи:

`spark-submit PySparkMLFit.py train.parquet test.parquet`
#или
`python PySparkMLFit.py train.parquet test.parquet`

где:
* **train.parquet** - путь к датасету, который необходимо использовать для обучения (скачать)
* **test.parquet** - путь к датасету, который необходимо использовать для оценки полученной модели (скачать)

2) PySparkMLPredict.py - задача, которая должна загружать модель и строить предсказание над переданными ей данными.
Варианты запуска задачи:

`spark-submit PySparkMLPredict.py test.parquet result`
#или
`python PySparkMLPredict.py test.parquet result`

где:
* **test.parquet** - путь к датасету, на основе данных которого нужно выполнить предсказания CTR (скачать)
* **result** - путь, по которому будет сохранен результат предсказаний в формате CSV следующего вида [ad_id, prediction]

Пожалуйста, используйте Pipeline.

