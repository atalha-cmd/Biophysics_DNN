# A makefile for screened Coulombic potential 
CC = gfortran 
FLAGS = -c 
HEADERS = 
SOURCES = modules.f90 main.f90 treecode.f90
MODULES = comdata.mod treecode_procedures.mod
OBJECTS = modules.o treecode.o main.o  
EXECUTABLE = a.out 
$(EXECUTABLE): $(MODULES) $(OBJECTS)
	$(CC) -o $(EXECUTABLE) *.o 
comdata.mod: modules.f90
	$(CC) $(FLAGS) modules.f90
treecode_procedures.mod: treecode.f90
	$(CC) $(FLAGS) treecode.f90
treecode.o: treecode.f90
	$(CC) $(FLAGS) treecode.f90		
main.o: main.f90
	$(CC) $(FLAGS) main.f90
clean:
	-rm -f $(OBJECTS) $(EXECUTABLE) $(MODULES) 
