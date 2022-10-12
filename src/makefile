CXX=g++
CPPFLAGS=-lstdc++ -std=c++17
WCHECK=-Wall -Wextra -Werror
TESTFLAG=-lgtest -pthread --coverage -fprofile-arcs -ftest-coverage
DEBUGFLAG=-ggdb3
TEXI2DVI=makeinfo --html
TESTFILE=model/tests/test.cpp
MODEL=model/matrix_perceptron/matrix_perceptron.cpp \
	model/graph_perceptron/graph_perceptron.cpp \
	model/matrix_perceptron/matrix.cpp \
	model/samples_dataset.cpp \
	model/perceptron.cpp \
	model/model.cpp

INSTALL_DIR=../install/
BUILD_DIR=build/
TAR_DIR=mlp_tar/

.PHONY: all install uninstall clean dvi dist tests debug leaks cpplint cppcheck mlp

all: mlp

install: mlp
	@mkdir $(INSTALL_DIR)
	@cp $(BUILD_DIR)network.app/Contents/MacOS/network $(INSTALL_DIR)mlp
	@make dvi
	@cp mlp.html/index.html $(INSTALL_DIR)index.html
	@make clean_build

uninstall:
	@rm -rf $(INSTALL_DIR)

dvi:
	@$(TEXI2DVI) mlp.texi

dist:
	@mkdir $(TAR_DIR)
	@cp makefile mlp.texi network.pro main.cpp Reaserch.md $(TAR_DIR)
	@cp -r controller icons icons images_for_research model sources view $(TAR_DIR)
	@tar cf mlp.tar $(TAR_DIR)
	@rm -rf $(TAR_DIR)

tests:
	@$(CXX) $(TESTFILE) $(MODEL) $(CPPFLAGS) $(DEBUGFLAG) $(TESTFLAG) -o test.out
	@./test.out
	make clean

mlp:
	@qmake -o $(BUILD_DIR)/makefile
	@make -C $(BUILD_DIR)
	@cp $(BUILD_DIR)/network.app/Contents/MacOS/network mlp

clean:
	@rm -rf test *.gch *.o *.a *.user
	@rm -f *.o *.a *.gcno *.gcda *.out test *.gch 
	@rm -R -f ../.DS_Store .DS_Store *.dSYM mlp
	@rm -rf headers/*.h.gch
	@rm -rf tests/*.o CPPLINT.cfg
	@rm -rf view/*.user *.gif 
	@rm -rf .DS_Store DS_Store *.tmp
	@rm -rf mlp.html/ mlp_tar/ mlp.tar ../build*


clean_build: clean
	@make -C $(BUILD_DIR) clean
	@rm -rf *.qrc .qmake.stash mlp
	@rm -rf $(BUILD_DIR) *.tar 

leaks: debug
	@CK_FORK=no leaks --atExit -- ./debug.out

debug: clean
	@$(CXX) $(TESTFILE) $(MODEL) $(CPPFLAGS) $(DEBUGFLAG) $(WCHECK) $(TESTFLAG) -o debug.out

linter:
	@cp ../materials/linters/CPPLINT.cfg .
	@python3 ../materials/linters/cpplint.py --extensions=cpp model/*.h model/*.cpp view/*.cpp view/*.h *.cpp *.h model/graph_perceptron/*.cpp model/graph_perceptron/*.h model/tests/*.cpp model/matrix_perceptron/*.cpp model/matrix_perceptron/*.h
	@rm CPPLINT.cfg

cppcheck:
	@cppcheck --enable=all --suppress=missingIncludeSystem model/*.h model/*.cpp view/*.cpp view/*.h *.cpp *.h model/graph_perceptron/*.cpp model/graph_perceptron/*.h model/tests/*.cpp model/matrix_perceptron/*.cpp model/matrix_perceptron/*.h
