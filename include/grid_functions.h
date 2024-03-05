#ifndef GRID_FUNCTIONS_H
#define GRID_FUNCTIONS_H

#include <string>
#include <sstream>

void fill_data_in_grid_default_way(unsigned *&grid_ptr, long sizeofgrid);
void fill_data_in_grid_from_inputfile(unsigned *&grid_ptr, long sizeofgrid, const std::string& filename);
void dump_data_from_grid_into_outputfile(unsigned *&grid_ptr, long sizeofgrid, const std::string& filename);
void print_data_in_grid_default_way(unsigned *&grid_ptr, long sizeofgrid, const std::string &msg);

#endif // GRID_FUNCTIONS_H