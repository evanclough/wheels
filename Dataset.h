#include <vector>
#include <memory>
#include <string>

//the dataset class provides stores some feature and label data, and provides
//some standard methods for accessing/editing them.
class Dataset {
    private:
        std::unique_ptr<std::vector<std::vector<float>>> feature_data;
        std::unique_ptr<std::vector<std::string>> feature_names;
        std::unique_ptr<std::vector<float>> label_data;
        int dataset_size, num_features;
    public:

        //constructors
        //from direct data
        Dataset(std::vector<std::vector<float>> feature_data, std::vector<std::string> feature_names, std::vector<float> label_data);

        //from csv with specified feature and label columns
        Dataset(std::string filename, std::vector<std::string> feature_columns, std::string label_column);

        //getters and setters
        std::vector<std::vector<float>> get_feature_data();
        std::vector<std::string> get_feature_names();
        std::vector<float> get_label_data();
        int get_dataset_size();
        int get_num_features();
        
        //utilities

        //adds data pair
        void add_data_pair(std::vector<float> features, float label);

        //adds list of data pairs
        void add_data_pairs(std::vector<std::vector<float>> features, std::vector<float> labels);

        //removes pair of data given index
        void remove_data_pair(int index);

        //returns zipped set of feature and label data
        std::vector<std::vector<std::vector<float>>> zip_data();
        
        //shuffles index
        void shuffle_dataset();
        
};