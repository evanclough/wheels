#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <memory>
#include <iostream>
#include <fstream>
#include <iterator>

#include "Utilities.h"
#include "Dataset.h"

//basic constructor just takes in raw data and stores it in instance
Dataset::Dataset(std::vector<std::vector<float>> feature_data, std::vector<std::string> feature_names, std::vector<std::vector<float>> label_data) {
    //first check if number of feature and label entries are same, throw error if not
    if(feature_data.size() != label_data.size()){
        throw std::invalid_argument("feature and label sets must be same size.");
    }

    //must be initialized with some data.
    if(feature_data.size() == 0){
        throw std::invalid_argument("Dataset must be provided some data to be initialized.");   
    }

    //check if all feature data entries are of same dimension, throw error if not.
    for(int i = 1; i < feature_data.size(); i++){
        if (feature_data[i].size() != feature_data[0].size()) {
            throw std::invalid_argument("feature data must be of uniform dimension.");
        }
    }
    
    //check if all label data entries are of same dimension, throw error if not
    for(int i = 1; i < label_data.size(); i++){
        if(label_data[i].size() != label_data[0].size()) {
            throw std::invalid_argument("label data must be of uniform dimension.");
        }
    }

    this->num_features = feature_data[0].size();
    this->num_labels = label_data[0].size();

    //check if feature name array passed in is either empty, or proper size.
    if(feature_names.size() == 0){
        std::vector<std::string> temp = {};
        this->feature_names = std::make_unique<std::vector<std::string>>(temp);
        for(int i = 0; i < this->num_features; i++){
            this->feature_names->push_back(std::to_string(i));
        }   
    }else if(feature_names.size() == this->num_features){
        this->feature_names = std::make_unique<std::vector<std::string>>(feature_names);
    }else{
        throw std::invalid_argument("feature_names array must be either empty or appropriate size according to num_features");
    }


    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(feature_data);
    this->label_data = std::make_unique<std::vector<std::vector<float>>>(label_data);
    this->dataset_size = feature_data.size();
}

//constructor from CSV
Dataset::Dataset(std::string filename, std::vector<std::string> feature_columns, std::vector<std::string> label_columns){
    int num_features = feature_columns.size();
    int num_labels = label_columns.size();
    //check feature_columns isn't empty, throw error if it is
    if(num_features == 0){
        throw std::invalid_argument("You must specify at least one column of features for the dataset.");
    }

    if(num_labels == 0){
        throw std::invalid_argument("You must specify at least one column of labels for the dataset.");
    }

    std::vector<std::vector<float>> feature_data;
    std::vector<std::vector<float>> label_data;

    // open the csv file
    std::ifstream csv(filename);

    //check file actually got opened, throw error if not
    if(!csv.is_open()){
        throw std::invalid_argument("Couldn't open specified CSV.");
    }

    std::string current_line;
    //parse column names from first line and trim them
    std::getline(csv, current_line);
    std::vector<std::string> column_names = Utilities::tokenize(current_line);
    for(int i = 0; i < column_names.size(); i++){
        column_names[i] = Utilities::trim(column_names[i]);
        std::cout << column_names[i] << " " << column_names[i].size() << std::endl;
    }

    //create and populatre array with order of columns to pull from csv according to feature and label names specified in input
    std::vector<int> feature_pull_order;
    std::vector<int> label_pull_order;
    for(int i = 0; i < num_features; i++){
        //iterator of elemenet, if its .end(), feature not found, throw error
        std::cout << feature_columns[i] << " " << feature_columns[i].size() << std::endl;
        auto it = std::find(column_names.begin(), column_names.end(), feature_columns[i]);
        if(it == column_names.end()){
            throw std::invalid_argument("Specified column to pull not found in specified CSV.");
        }
        feature_pull_order.push_back(it - column_names.begin());
    }
    //same thing for label columns
    for(int i = 0; i < num_labels; i++){
        //iterator of elemenet, if its .end(), label not found, throw error
        std::cout << label_columns[i] << " " << label_columns[i].size() << std::endl;
        auto it = std::find(column_names.begin(), column_names.end(), label_columns[i]);
        if(it == column_names.end()){
            throw std::invalid_argument("Specified column to pull not found in specified CSV.");
        }
        label_pull_order.push_back(it - column_names.begin());
    }

    //go through csv, tokenize and trim line, set data arrays according to order we made earlier
    while(std::getline(csv, current_line)){
        std::vector<std::string> tokens = Utilities::tokenize(current_line);
        for(int i = 0; i < tokens.size(); i++){
            tokens[i] = Utilities::trim(tokens[i]);
        }
        //check tokens array is of proper length,. throw error if not
        if(tokens.size() != column_names.size()){
            throw std::invalid_argument("CSV doesn't have uniform number of entries in each row.");
        }
        std::vector<float> features;
        std::vector<float> labels;
        for(int i = 0; i < feature_pull_order.size(); i++){
             features.push_back(std::stof(tokens[feature_pull_order[i]]));
        }
        for(int i = 0; i < label_pull_order.size(); i++){
             labels.push_back(std::stof(tokens[label_pull_order[i]]));
        }

        feature_data.push_back(features);
        label_data.push_back(labels);
    }
    //once finished reading csv data, close file, and populate fields of class with info
    csv.close();
    this->num_features = num_features;
    this->feature_names = std::make_unique<std::vector<std::string>>(feature_columns);
    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(feature_data);
    this->label_data = std::make_unique<std::vector<std::vector<float>>>(label_data);
    this->dataset_size = feature_data.size();
}

//copy constructor
Dataset::Dataset(const Dataset& cpy){
    this->num_features = cpy.num_features;
    this->num_labels = cpy.num_labels;
    this->dataset_size = cpy.dataset_size;
    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(*(cpy.feature_data));
    this->feature_names = std::make_unique<std::vector<std::string>>(*(cpy.feature_names));
    this->label_data = std::make_unique<std::vector<std::vector<float>>>(*(cpy.label_data));
}

//constructs dataset from ubyte file, used MNIST data
Dataset::Dataset(std::string feature_data_filename, std::string label_data_filename){
	//initialize data vectors
	std::vector<std::vector<float>> feature_data, label_data;

	//open ubyte files
	std::ifstream feature_data_file(feature_data_filename, std::ios::binary), label_data_file(label_data_filename, std::ios::binary);

	//check if files successfully opened, if not, throw error
	if(!feature_data_file.good()){
		throw std::invalid_argument("feature data file did not successfully open.");
	}

	if(!label_data_file.good()){
		throw std::invalid_argument("label data file did not successfully open.");
	}

	//copy into vectors

	feature_data_file.unsetf(std::ios::skipws);	
	label_data_file.unsetf(std::ios::skipws);	
	std::streampos feature_data_file_size, label_data_file_size;
	
	feature_data_file.seekg(0, std::ios::end);
	feature_data_file_size = feature_data_file.tellg();
	feature_data_file.seekg(0, std::ios::beg);

	label_data_file.seekg(0, std::ios::end);
	label_data_file_size = label_data_file.tellg();
	label_data_file.seekg(0, std::ios::beg);

	std::vector<uint8_t> feature_data_bytes, label_data_bytes;
	feature_data_bytes.reserve(feature_data_file_size);
	label_data_bytes.reserve(label_data_file_size);
	
	feature_data_bytes.insert(feature_data_bytes.begin(), std::istream_iterator<uint8_t>(feature_data_file), std::istream_iterator<uint8_t>());
	label_data_bytes.insert(label_data_bytes.begin(), std::istream_iterator<uint8_t>(label_data_file), std::istream_iterator<uint8_t>());


	//close files
	feature_data_file.close();
	label_data_file.close();

	//push image files one by one into feature data array
	//start at index 16 to skip header
	//images are 28 by 28 so total size for each feature is 784
	for(int i = 0; i < (feature_data_bytes.size() - 16) / 784; i++){
		feature_data.push_back({});
		for(int j = 0; j < 784; j++){
			feature_data[i].push_back(feature_data_bytes[16 + (i * 784) + j]);
		}
	}

	//create label data as array of onehot vectors with associated digit as the hot number
	// start at index 8 to skip header
	for(int i = 0; i < feature_data.size(); i++){
		std::vector<float> temp(10, 0);
		temp[label_data_bytes[8 + i]] = 1;
		label_data.push_back(temp);
	}

	this->num_features = feature_data[0].size();
	this->num_labels = label_data[0].size();

	std::vector<std::string> feature_names;
	for(int i = 0; i < num_features; i++){
		feature_names.push_back(std::to_string(i));
	}

	this->feature_names = std::make_unique<std::vector<std::string>>(feature_names);	
    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(feature_data);
    this->label_data = std::make_unique<std::vector<std::vector<float>>>(label_data);
	this->dataset_size = feature_data.size();
}

//constructs dataset from ubyte file, used MNIST data, only load in first n images n
Dataset::Dataset(std::string feature_data_filename, std::string label_data_filename, int n){
	//initialize data vectors
	std::vector<std::vector<float>> feature_data, label_data;

	//open ubyte files
	std::ifstream feature_data_file(feature_data_filename, std::ios::binary), label_data_file(label_data_filename, std::ios::binary);

	//check if files successfully opened, if not, throw error
	if(!feature_data_file.good()){
		throw std::invalid_argument("feature data file did not successfully open.");
	}

	if(!label_data_file.good()){
		throw std::invalid_argument("label data file did not successfully open.");
	}

	//copy into vectors

	feature_data_file.unsetf(std::ios::skipws);	
	label_data_file.unsetf(std::ios::skipws);	
	std::streampos feature_data_file_size, label_data_file_size;
	
	feature_data_file.seekg(0, std::ios::end);
	feature_data_file_size = feature_data_file.tellg();
	feature_data_file.seekg(0, std::ios::beg);

	label_data_file.seekg(0, std::ios::end);
	label_data_file_size = label_data_file.tellg();
	label_data_file.seekg(0, std::ios::beg);

	std::vector<uint8_t> feature_data_bytes, label_data_bytes;
	feature_data_bytes.reserve(feature_data_file_size);
	label_data_bytes.reserve(label_data_file_size);
	
	feature_data_bytes.insert(feature_data_bytes.begin(), std::istream_iterator<uint8_t>(feature_data_file), std::istream_iterator<uint8_t>());
	label_data_bytes.insert(label_data_bytes.begin(), std::istream_iterator<uint8_t>(label_data_file), std::istream_iterator<uint8_t>());

	//close files
	feature_data_file.close();
	label_data_file.close();


	//push image files one by one into feature data array
	//start at index 16 to skip header
	//images are 28 by 28 so total size for each feature is 784
	for(int i = 0; i < n; i++){
		feature_data.push_back({});
		for(int j = 0; j < 784; j++){
			feature_data[i].push_back(feature_data_bytes[16 + (i * 784) + j]);
		}
	}


	//create label data as array of onehot vectors with associated digit as the hot number
	// start at index 8 to skip header
	for(int i = 0; i < n; i++){
		std::vector<float> temp(10, 0);
		temp[label_data_bytes[8 + i]] = 1;
		label_data.push_back(temp);
	}

	this->num_features = feature_data[0].size();
	this->num_labels = label_data[0].size();

	std::vector<std::string> feature_names;
	for(int i = 0; i < num_features; i++){
		feature_names.push_back(std::to_string(i));
	}

	this->feature_names = std::make_unique<std::vector<std::string>>(feature_names);	
    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(feature_data);
    this->label_data = std::make_unique<std::vector<std::vector<float>>>(label_data);
	this->dataset_size = feature_data.size();
}

//getters and setters
std::vector<std::vector<float>> Dataset::get_feature_data(){
    return *(this->feature_data);
}

std::vector<std::string> Dataset::get_feature_names(){
    return *(this->feature_names);
}

std::vector<std::vector<float>> Dataset::get_label_data(){
    return *(this->label_data);
}

int Dataset::get_dataset_size() {
    return this->feature_data->size();
}

int Dataset::get_num_features() {
    return this->num_features;
}

int Dataset::get_num_labels() {
    return this->num_labels;
}

//adds feature-label pair to dataset
void Dataset::add_data_pair(std::vector<float> features, std::vector<float> labels){
    //check if feature is of appropriate dimension, throw error if not
    if(features.size() != this->num_features){
        throw std::invalid_argument("Dimension of passed feature doesn't match that of dataset.");
    }

    if(labels.size() != this->num_labels){
        throw std::invalid_argument("Dimension of passed label doesn't match that of dataset.");
    }

    //adjust dataset size accordingly and add new data

    this->dataset_size++;

    this->feature_data->push_back(features);
    this->label_data->push_back(labels);
}

//adds multiple feature-label pairs to dataset
void Dataset::add_data_pairs(std::vector<std::vector<float>> features, std::vector<std::vector<float>> labels){
    //check features are all of right size
    for(int i = 0; i < features.size(); i++){
        if(features[i].size() != this->num_features){
            throw std::invalid_argument("All passed in feature sets must be matching dimension of dataset.");
        }
    }

    //check that number of features and labels matches 
    for(int i = 0; i < labels.size(); i++){
        if(labels[i].size() != this->num_labels){
            throw std::invalid_argument("all passed in label sets must be matching dimension of dataset.");   
        }
    }

    //check that number of features and labels matches 
    if(features.size() != labels.size()){
        throw std::invalid_argument("number of feature sets to add to dataset must be same as number of labels.");
    }

    //add data
    for(int i = 0; i < features.size(); i++){
        this->add_data_pair(features[i], labels[i]);
    }

    //adjust dataset size accordingly

    this->dataset_size += features.size();
}

//removes feature-label pair given index
void Dataset::remove_data_pair(int index){
    //check if index is in bounds, throw error if not
    if(index < 0 || index >= this->dataset_size){
        throw std::invalid_argument("Index out of range.");
    }

    this->feature_data->erase(this->feature_data->begin() + index);
    this->label_data->erase(this->label_data->begin() + index);
    this->dataset_size--;
}

//returns zipped feature and label data
std::vector<std::vector<std::vector<float>>> Dataset::zip_data(){
    std::vector<std::vector<std::vector<float>>> zipped_data = {};
    for(int i = 0; i < this->dataset_size; i++){
        zipped_data.push_back({this->feature_data->at(i), this->label_data->at(i)});
    }
    return zipped_data;
}

//shuffles dataset
//todo: make this better
void Dataset::shuffle_dataset(){
    //first zip up features and labels

    int original_dataset_size = this->dataset_size;
    std::vector<std::vector<std::vector<float>>> zipped_data = this->zip_data();

    //initalize random generator we'll be using
    std::random_device rd;
    std::mt19937 gen(rd());
    //shuffle zipped array
    std::shuffle(zipped_data.begin(), zipped_data.end(), gen);
    //put back into base arrays
    this->feature_data->clear();
    this->label_data->clear();
    this->dataset_size = 0;
    for(int i = 0; i < original_dataset_size; i++){
        this->add_data_pair(zipped_data[i][0], zipped_data[i][1]);
    }
}


