# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../src/libidaten" -I"../src/multigputest" -I"../3rdparty/glm" -I"../3rdparty/imgui" 
Release_Include_Path=-I"../src/libaten" -I"../src/libatenscene" -I"../src/libidaten" -I"../src/multigputest" -I"../3rdparty/glm" -I"../3rdparty/imgui" 

# Library paths...
Debug_Library_Path=-L"x64/Debug" 
Release_Library_Path=-L"x64/Release" 

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lidaten -lcuda -lcudart -lcudadevrt -fopenmp -lGL -lglfw -lGLEW  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -laten -latenscene -lidaten -lcuda -lcudart -lcudadevrt -fopenmp -lGL -lglfw -lGLEW  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D __AT_DEBUG__ 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g -std=c++11 -fopenmp 
Release_Compiler_Flags=-O2 -std=c++11 -fopenmp 

# Location of the CUDA Toolkit
CUDA_PATH?="/usr/local/cuda"
NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CPP_COMPILER)

Debug_Include_Path+=-I"/usr/local/cuda/include"
Release_Include_Path+=-I"/usr/local/cuda/include"

Debug_Library_Path+=-L"/usr/local/cuda/lib64"
Release_Library_Path+=-L"/usr/local/cuda/lib64"

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders x64/Debug/multigputest/src/common/scenedefs.o x64/Debug/multigputest/src/multigputest/main.o 
	g++ x64/Debug/multigputest/src/common/scenedefs.o x64/Debug/multigputest/src/multigputest/main.o  $(Debug_Library_Path) $(Debug_Libraries) -Wl,-rpath,./ -o ../src/multigputest/multigputest.exe

# Compiles file ../src/common/scenedefs.cpp for the Debug configuration...
-include x64/Debug/multigputest/src/common/scenedefs.d
x64/Debug/multigputest/src/common/scenedefs.o: ../src/common/scenedefs.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/common/scenedefs.cpp $(Debug_Include_Path) -o x64/Debug/multigputest/src/common/scenedefs.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/common/scenedefs.cpp $(Debug_Include_Path) > x64/Debug/multigputest/src/common/scenedefs.d

# Compiles file ../src/multigputest/main.cpp for the Debug configuration...
-include x64/Debug/multigputest/src/multigputest/main.d
x64/Debug/multigputest/src/multigputest/main.o: ../src/multigputest/main.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/multigputest/main.cpp $(Debug_Include_Path) -o x64/Debug/multigputest/src/multigputest/main.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/multigputest/main.cpp $(Debug_Include_Path) > x64/Debug/multigputest/src/multigputest/main.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/multigputest/src/common/scenedefs.o x64/Release/multigputest/src/multigputest/main.o 
	g++ x64/Release/multigputest/src/common/scenedefs.o x64/Release/multigputest/src/multigputest/main.o  $(Release_Library_Path) $(Release_Libraries) -Wl,-rpath,./ -o ../src/multigputest/multigputest.exe

# Compiles file ../src/common/scenedefs.cpp for the Release configuration...
-include x64/Release/multigputest/src/common/scenedefs.d
x64/Release/multigputest/src/common/scenedefs.o: ../src/common/scenedefs.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/common/scenedefs.cpp $(Release_Include_Path) -o x64/Release/multigputest/src/common/scenedefs.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/common/scenedefs.cpp $(Release_Include_Path) > x64/Release/multigputest/src/common/scenedefs.d

# Compiles file ../src/multigputest/main.cpp for the Release configuration...
-include x64/Release/multigputest/src/multigputest/main.d
x64/Release/multigputest/src/multigputest/main.o: ../src/multigputest/main.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/multigputest/main.cpp $(Release_Include_Path) -o x64/Release/multigputest/src/multigputest/main.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/multigputest/main.cpp $(Release_Include_Path) > x64/Release/multigputest/src/multigputest/main.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/multigputest/src/common
	mkdir -p x64/Debug/multigputest/src/multigputest
	mkdir -p ../src/multigputest
	mkdir -p x64/Release/multigputest/src/common
	mkdir -p x64/Release/multigputest/src/multigputest
	mkdir -p ../src/multigputest

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/multigputest/src/common/*.o
	rm -f x64/Debug/multigputest/src/common/*.d
	rm -f x64/Debug/multigputest/src/multigputest/*.o
	rm -f x64/Debug/multigputest/src/multigputest/*.d
	rm -f x64/Debug/multigputest/*.o
	rm -f x64/Debug/multigputest/*.d
	rm -f ../src/multigputest/multigputest.exe
	rm -f x64/Release/multigputest/src/common/*.o
	rm -f x64/Release/multigputest/src/common/*.d
	rm -f x64/Release/multigputest/src/multigputest/*.o
	rm -f x64/Release/multigputest/src/multigputest/*.d
	rm -f x64/Release/multigputest/*.o
	rm -f x64/Release/multigputest/*.d
	rm -f ../src/multigputest/multigputest.exe

