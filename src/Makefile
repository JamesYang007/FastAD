CXX = g++ -std=c++11
INCLUDE = -I../include/
LDLIBS = 
CXXFLAGS = -O2 $(INCLUDE)
LDFLAGS = 

DEP = test.h ../include/*.h
CPP = main.cpp test.cpp

main: $(DEP) $(CPP)
	$(CXX) $(CXXFLAGS) $(CPP) $(LDFLAGS) $(LDLIBS) -o main


.PHONY: clean

clean:
	rm -rf main main.dSYM

.PHONY: all

all: clean main
