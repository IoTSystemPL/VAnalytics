#### Instalacja OpenCV 4.4.0 dla Visual Studio 2017:

##### Pobieranie:
- Pobierz źródła z github: https://github.com/opencv/opencv (4.5.3)
- Pobierz źródła z github: https://github.com/opencv/opencv_contrib (4.5.3)

##### CMake:
- Uruchom cmake-gui.exe
- W polu "Where is the source code:" podaj ścieżkę do źródeł OpenCV (np: C:/Users/Third-party/OpenCV)
- W polu "Where to build the binares:" podaj ścieżkę gdzie ma znaleźć się skompilowane OpenCV (np: C:/Users/Tools/opencv)
- Kliknij "Configure", z listy generatorów wybierz "Visual Studio 16 2019 Win64" albo "Unix makefiles" dla linuxa
- Żeby zbudować za pomocą C++17, dodaj nową zmienną o nazwie CMAKE_CXX_STANDARD, typie "string" i przypisz jej wartość 17.
- Na liście konfiguracji odznacz "With VTK"
- Na liście konfiguracji w polu "OPENCV_EXTRA_MODULES_PATH" podaj ścieżkę do źródeł OpenCV_Contribue (np. C:/Users/Tools/opencv_contrib-4.5.3_source/modules)
- Opcjonalnie zaznacz opcję BUILD_WITH_DEBUG_INFO, OPENCV_DNN_CUDA (ewentualnie WITH_CUDA)
- Kliknij "Configure" i jeżeli nic na liście nie jest na czerwono można kliknąć "Generate"

##### Kompilacja (Ten krok powinien być powtórzony dla wersji Debug i Release):
- Z katalogu wybranego w CMake "Where to build the binares", otwórz plik "OpenCV.sln" w VS 2019
- W oknie "Solution Explorer" kliknij prawym na "CMakeTargets/ALL_BUILD" i wybierz "Build"
- W oknie "Solution Explorer" kliknij prawym na "CMakeTargets/INSTALL" i wybierz "Build"
lub dla linuxa:
- Przejdź w terminalu do katalogu wybranego w CMake "Where to build the binares" i wykonaj "make -j7" a następnie "sudo make install"

##### Zmienne środowiskowe:
- Dodaj nową zmienną środowiskową o nazwie "OPENCV_DIR"
- Przypisz jej ścieżkę do skompilowanych bibliotek OpenCV (schemat: C:/sciezka_where_to_build_the_binares/install/x64/vc16)
- Do zmiennej PATH dodaj ";%OPENCV_DIR%\bin"
- Dodaj zmienną środowiskową "OPENCV_VER" z taką wartością jak wersja zainstalowanej biblioteki (np.dla wersji 4.5.3 ustaw 453).


#### Instalacja CUDA 11 dla Windows 64bit (pomiń jeżeli już masz zainstalowaną jakąś wersję CUDY)

##### Pobieranie
- Pobierz odpowiedni instalator (najlepiej wersję 'local') ze strony https://developer.nvidia.com/cuda-downloads

##### Instalacja
- Uruchom instalator

##### Zmienne środowiskowe
- Dodaj zmienną środowiskową o nazwie "CUDA_PATH" (o ile sama się nie dodała podczas instalacji)
- Przypisz jej ścieżkę do analogicznej lokalizacji jak "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4"


#### Instalacja OpenBlas

##### Automatyczne pobieranie
- W głównym folderze projektu uruchom skrypt "get_dependency.py" wpisując w wierszu poleceń "python get_dependency.py"
- Jeżeli brakuje Ci biblioteki "requests", możesz ją zainstalować ręcznie poleceniem "pip install requests"
- Jeżeli z jakiegoś powodu skrypt w ogóle nie zadziała, zastosuj się do poleceń z poniższej sekcji

##### Ręczne pobieranie
- Pobierz i rozpakuj http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-openblas-0.3.9-1-any.pkg.tar.xz
- Pobierz i rozpakuj http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-libgfortran-9.3.0-2-any.pkg.tar.xz
- Z rozpakowanych folderów wyciągnij podfoldery 'mingw64' i połącz je ze sobą

##### Zmienne środowiskowe
- Folder "mingw64" możesz przenieść w dowolne inne miejsce
- Dodaj zmienną środowiskową o nazwie "MINGW64_DIR"
- Przypisz jej ścieżkę do zawartości folderu "mingw64"
- Do zmiennej "PATH" dodaj "%MINGW64_DIR%\bin"

### Modele sieci neuronowych
- Z lokalizacji "\\ml_models" pobierz listę wytrenowanych sieci neuronowych.
