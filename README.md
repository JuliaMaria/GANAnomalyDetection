# GANAnomalyDetection

# Problemy podczas realizacji projektu

Nie udało nam się odtworzyć wyników z artykułu określonych jako wymagania minimalne, ponieważ napotkałyśmy następujące problemy:
- Złożona metoda preprocessingu danych jest w artykule jedynie wspomniana, natomiast nie jest dokładnie opisana i nie jest dostępny jej kod, przez co nie jest możliwe dokładnie takie samo przygotowanie danych w celu dokładnego odtworzenia wyników.
- Rozmiar zbiorów danych z artykułu jest bardzo duży, przez co nie jest możliwe wytrenowanie modelu na całym zbiorze danych przez wymaganą liczbę epok na sprzęcie, który mamy do dyspozycji. Z tego powodu musiałyśmy ograniczyć się jedynie do wycinku zbioru danych z artykułu oraz zmniejszyć liczbę epok treningu.
- Kod modelu dostępny na GitHubie napisany jest w TensorFlow 1, co spowodowałoby trudność w dodaniu transformera ze względu na brak dostępnej odpowiedniej warstwy w Kerasie. Z tego powodu zdecydowałyśmy się skorzystać z innej dostępnej implementacji tego samego modelu w PyTorch.
- Uzyskane wyniki dla tych danych i kodu są gorsze niż w artykule, jednak możemy potraktować je jako baseline, do którego można porównywać wyniki uzyskane po dodaniu transformera dla tych samych danych i kodu, żeby sprawdzić czy dodanie transformera przyczynia się do poprawy jakości działania.

# Wykonana praca

Wykonałyśmy następujące kroki:
- Dopasowałyśmy dostępną implementację modelu TadGAN napisaną w PyTorch, tak aby można było wykonać trening i odpytać model (model.py, training_lstm.ipynb)
- Przygotowałyśmy fragment datasetu 'Ambient temperature system failure', który był wykorzystywany przez autorów artykułu (folder datasets)
- Wykonałyśmy trening i ewaluację modelu na datasecie 'Ambient temperature System Failure', żeby zweryfikować czy wykorzystana implementacja modelu uczy się, wykresy porównawcze są dostępne w folderze results
- Zmodyfikowałyśmy model, żeby zamiast LSTM korzystał z warstwy transformer oraz wykonałyśmy trening i ewaluację tego modelu (model_transformer.py, training_transformer.ipynb)
- Okazało się, że po dodaniu transformera model nie jest w stanie odtworzyć żadnych zależności w strumieniu danych, po treningu odtwarza jedynie losowy szum

# Contents

Training TadGAN on 'Ambient temperature system failure':

datasets - folder with 'Ambient temperature System Failure' benchmarking dataset (data from: https://github.com/numenta/NAB/blob/master/data/realKnownCause/ambient_temperature_system_failure.csv)

model.py, interface.py, anomaly_detection.py, utils.py - code for model structure, training, anomaly detection and dataset utils functions for TadGAN model with LSTM (code from: https://github.com/arunppsg/TadGAN)

model_transformer.py - code for TadGAN model with transformer

training_lstm.ipynb - notebook with training pipeline and evaluation for TadGAN model with LSTM

results - plots showing original and reconstructed signal for TadGAN model with LSTM

training_transformer.ipynb - notebook with training pipeline and evaluation for TadGAN model with transformer

evaluation_statistics.ipynb - notebook with training and evaluation statistics for TadGAN models with LSTM and with transformer


