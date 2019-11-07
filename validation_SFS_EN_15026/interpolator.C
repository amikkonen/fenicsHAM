//https://fenicsproject.org/qa/12901/including-a-function-in-c-expression/
//https://answers.launchpad.net/dolfin/+question/227358


#include <vector>

class MyFunc : public Expression
{

private:
    // Material 0
    std::vector<double> val_table0;
    std::vector<double> x_table0;

    // Material 1
    std::vector<double> val_table1;
    std::vector<double> x_table1;

    // Material 2
    std::vector<double> val_table2;
    std::vector<double> x_table2;
    
    // Material 3
    std::vector<double> val_table3;
    std::vector<double> x_table3;


    void print_given_table(std::vector<double> &val_table,
                              std::vector<double> &x_table) {
        for (int k = 0; k < val_table.size(); ++k) {
            std::cout << x_table[k] << " " << val_table[k] << '\n';   
        }
    }

    double interpolate_material(const double xi, 
                       const std::vector<double> &val_table,
                       const std::vector<double> &x_table) const {

        double val = 0.0;
        
        // upper and lower
        if (xi <= x_table[0]) {
            val = val_table[0];
        } else if (xi >= x_table[val_table.size()-1]) {
            val = val_table[val_table.size()-1];
        } else {
            // Interpolation
            int k = 0;
            while (k < val_table.size()) {
                // Move to next slot if not this
                if (x_table[k] < xi) {
                    ++k;
                // Exact match
                } else if (x_table[k] == xi) {
                    val = val_table[k];        
                    break;
                // Interpolation from this slot
                } else {
                    double alpha = (xi-x_table[k-1])/(x_table[k]-x_table[k-1]);
                    val = alpha*(val_table[k]- val_table[k-1]) + val_table[k-1];
                    break;
                }
            }
        }

        return val;
    }


public:

    // Constractor for scalar  
    MyFunc() : Expression() {}

    // Members  
    std::shared_ptr<MeshFunction<std::size_t>> materials;
    std::shared_ptr<const Function> x_k; // NOTE! T is apperently a reserved variable or something.

    void push_table(const int matLabel, const double val, const double x) {
//        std::cout << "set one " << matLabel << " " << val << " " << phi << '\n';
        switch(matLabel) {
            case 0 : 
                val_table0.push_back(val);
                x_table0.push_back(x);
                break;   
            case 1 : 
                val_table1.push_back(val);
                x_table1.push_back(x);
                break;   
            case 2 : 
                val_table2.push_back(val);
                x_table2.push_back(x);
                break;   
            case 3 : 
                val_table3.push_back(val);
                x_table3.push_back(x);
                break;   
        }
    }  


    void print_table(const int matLabel) {
        

        switch(matLabel) {
            case 0 : 
                print_given_table(val_table0, x_table0);
                break;   
            case 1 : 
                print_given_table(val_table1, x_table1);
                break;   
            case 2 : 
                print_given_table(val_table2, x_table2);
                break;   
            case 3 : 
                print_given_table(val_table3, x_table3);
                break;   
        }
    }

    double interpolate_linear(const int matLabel, const double xi) const {
        double val = 0.0;
        switch(matLabel) {
            case 0 : 
                val = interpolate_material(xi, val_table0, x_table0);
                break;   
            case 1 : 
                val = interpolate_material(xi, val_table1, x_table1);
                break;   
            case 2 : 
                val = interpolate_material(xi, val_table2, x_table2);
                break;   
            case 3 : 
                val = interpolate_material(xi, val_table3, x_table3);
                break;   
        }

        return val;

    }


//     // Override eval at cell  
//    void eval(Array<double>& values,
//              const Array<double>& x,
//              const ufc::cell& cell) const {
////        std::cout << "HERE!" << std::endl;
//        x_k->eval(values, x);
////        std::cout << "HERE 2!" << std::endl;
//        const double x_val = values[0];
//  
//        values[0] = interpolate_linear((*materials)[cell.index], x_val);

//    }

    // Override eval at cell  
    void eval(Array<double>& values,
              const Array<double>& x,
              const ufc::cell& c) const {
//        std::cout << "HERE!" << std::endl;
        x_k->eval(values, x);
//        std::cout << "HERE 2!" << std::endl;
        const double x_val = values[0];
  
        assert(materials);
        const Cell celli(*materials->mesh(), c.index);
  
        values[0] = interpolate_linear((*materials)[celli.index()], x_val);

    }

};
