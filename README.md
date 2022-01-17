# GANAnomalyDetection

# Problemy podczas realizacji projektu

Nie udało nam się odtworzyć wyników z artykułu określonych jako wymagania minimalne, ponieważ napotkaliśmy następujące problemy:
- Złożona metoda preprocessingu danych jest w artykule jedynie wspomniana, natomiast nie jest dokładnie opisana i nie jest dostępny jej kod, przez co nie jest możliwe dokładnie takie samo przygotowanie danych w celu dokładnego odtworzenia wyników.
- Rozmiar zbiorów danych z artykułu jest bardzo duży, przez co nie jest możliwe wytrenowanie modelu na całym zbiorze danych przez wymaganą liczbę epok na sprzęcie, który mamy do dyspozycji. Z tego powodu musieliśmy ograniczyć się jedynie do wycinku zbioru danych NASA oraz zminiejszyć liczbę epok treningu.
- Kod modelu dostępny na GitHubie napisany jest w TensorFlow 1, co spowodowałoby trudność w dodaniu transformera ze względu na brak dostępnej odpowiedniej warstwy w Kerasie. Z tego powodu zdecydowaliśmy się skorzystać z innej dostępnej implementacji tego samego modelu w PyTorch (https://github.com/arunppsg/TadGAN).
- Uzyskane benchmarkowe wyniki dla tych danych i kodu są znacznie gorsze niż w artykule, jednak możemy potraktować je jako baseline, do którego będziemy porównywać wyniki uzyskane po dodaniu transformera dla tych samych danych i kodu, żeby sprawdzić czy dodanie transformera przyczyni się do poprawy jakości działania.

# Wykonana praca

Podsumowując dotychczasową pracę wykonałyśmy następujące kroki:
- Dopasowałyśmy dostępną implementację modelu TadGAN napisaną w PyTorch, tak aby można było wykonać trening i odpytać model (folder model_training)
- Przygotowałyśmy fragmentu datasetu NASA, który był wykorzystywany przez autorów artykułu (Convert_NASA_Data_to_Orion_Format.ipynb, PrepareData.ipynb)
- Wykonałyśmy trening modelu na datasecie 'Ambient temperature System Failure' (https://github.com/numenta/NAB/blob/master/data/realKnownCause/ambient_temperature_system_failure.csv), żeby zweryfikować czy wykorzystana implementacja modelu uczy się. Wykresy są dostępne w folderze count_reconstructed_error/results

# Contents

BenchmarkingResults.ipynb - notebook with summary of the benchmarking performance on fragment of NASA dataset

benchmarking_results - folder with results of the benchmarking performance on fragment of NASA dataset

model_training - folder with code for model training, anomaly detection and main pipeline (code from: https://github.com/arunppsg/TadGAN)

models - folder with trained models

Convert_NASA_Data_to_Orion_Format.ipynb - notebook for downloading NASA dataset (code from: https://github.com/sintel-dev/Orion)

PrepareData.ipynb - notebook for converting fragment of NASA dataset to the appropriate format

prepared_data - folder with fragment of NASA dataset for evaluation

labels.csv - folder with anomaly labels for NASA dataset

count_reconstructed_error - folder with training TadGAN on 'Ambient temperature system failure'

count_reconstructed_error/train_temperature - notebook with training pipeline and evaluation

count_reconstructed_error/results - plots showing original and reconstructed signal


