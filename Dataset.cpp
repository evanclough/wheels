#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <memory>
#include <iostream>
#include <fstream>

#include "Dataset.h"

//basic constructor just takes in raw data and stores it in instance
Dataset::Dataset(std::vector<std::vector<float>> feature_data, std::vector<std::string> feature_names, std::vector<float> label_data) {
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
    
    this->num_features = feature_data[0].size();

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
    this->label_data = std::make_unique<std::vector<float>>(label_data);
    this->dataset_size = this->feature_data->size();
}

Dataset::Dataset(std::string filename, std::vector<std::string> feature_columns, std::string label_column){
    int num_features = feature_columns.size();
 
    //check feature_columns isn't empty, throw error if it is
    if(num_features == 0){
        throw std::invalid_argument("You must specify at least one column of features for the dataset.");
    }

    std::vector<std::vector<float>> feature_data;
    std::vector<float> label_data;

    // open the csv file
    std::ifstream csv(filename);

    //check file actually got opened, throw error if not
    if(!csv.is_open()){
        throw std::invalid_argument("Couldn't open specified CSV.");
    }

    std::string current_line;
    //parse column names from first line
    std::vector<std::string> column_names;
    std::getline(csv, current_line);


    std::size_t index;
    while((index = current_line.find(",")) != std::string::npos){
        column_names.push_back(current_line.substr(0, index));
        current_line.erase(0, index + 1);
    } 
    column_names.push_back(current_line);

    //create and populatre array with order of columns to pull from csv according to feature and label names specified in input
    std::vector<int> feature_pull_order;
    int label_col_index;
    for(int i = 0; i < num_features; i++){
        //iterator of elemenet, if its .end(), feature not found, throw error
        auto it = std::find(column_names.begin(), column_names.end(), feature_columns[i]);
        if(it == column_names.end()){
            throw std::invalid_argument("Specified column to pull not found in specified CSV.");
        }
        feature_pull_order.push_back(it - column_names.begin());
    }

    //do same for label column
    auto it = std::find(column_names.begin(), column_names.end(), label_column);
    if(it == column_names.end()){
        throw std::invalid_argument("Specified column to pull not found in specified CSV.");
    }
    label_col_index = it - column_names.begin();
    //go through csv, tokenize line, set data arrays according to order we made earlier
    while(std::getline(csv, current_line)){
        std::vector<std::string> tokens;
        std::size_t index;
        while((index = current_line.find(",")) != std::string::npos){
            tokens.push_back(current_line.substr(0, index));
            current_line.erase(0, index + 1);
        } 
        tokens.push_back(current_line);
        //check tokens array is of proper length,. throw error if not
        if(tokens.size() != column_names.size()){
            throw std::invalid_argument("CSV doesn't have uniform number of entries in each row.");
        }
        std::vector<float> features;
        for(int i = 0; i < feature_pull_order.size(); i++){
             features.push_back(std::stof(tokens[feature_pull_order[i]]));
        }
        float label = std::stof(tokens[label_col_index]);

        feature_data.push_back(features);
        label_data.push_back(label);
    }
    //once finished reading csv data, close file, and populate fields of class with info
    csv.close();
    this->num_features = num_features;
    this->feature_names = std::make_unique<std::vector<std::string>>(feature_columns);
    this->feature_data = std::make_unique<std::vector<std::vector<float>>>(feature_data);
    this->label_data = std::make_unique<std::vector<float>>(label_data);
    this->dataset_size = this->feature_data->size();
}

//getters and setters
std::vector<std::vector<float>> Dataset::get_feature_data(){
    return *(this->feature_data);
}

std::vector<std::string> Dataset::get_feature_names(){
    return *(this->feature_names);
}

std::vector<float> Dataset::get_label_data(){
    return *(this->label_data);
}

int Dataset::get_dataset_size() {
    return this->dataset_size;
}

int Dataset::get_num_features() {
    return this->num_features;
}

//adds feature-label pair to dataset
void Dataset::add_data_pair(std::vector<float> features, float label){
    //check if feature is of appropriate dimension, throw error if not
    if(features.size() != this->num_features){
        throw std::invalid_argument("Dimension of passed feature doesn't match that of dataset.");
    }

    this->feature_data->push_back(features);
    this->label_data->push_back(label);
}

//adds multiple feature-label pairs to dataset
void Dataset::add_data_pairs(std::vector<std::vector<float>> features, std::vector<float> labels){
    //check features are all of right size
    for(int i = 0; i < features.size(); i++){
        if(features[i].size() != this->num_features){
            throw std::invalid_argument("All passed in feature sets must be matching dimension of dataset.");
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
        zipped_data.push_back({this->feature_data->at(i), {this->label_data->at(i)}});
    }
    return zipped_data;
}

//shuffles dataset
//todo: make this better
void Dataset::shuffle_dataset(){
    //first zip up features and labels
    std::vector<std::vector<std::vector<float>>> zipped_data = this->zip_data();

    //initalize random generator we'll be using
    std::random_device rd;
    std::mt19937 gen(rd());
    //shuffle zipped array
    std::shuffle(zipped_data.begin(), zipped_data.end(), gen);
    //put back into base arrays
    this->feature_data->clear();
    this->label_data->clear();
    for(int i = 0; i < this->dataset_size; i++){
        this->add_data_pair(zipped_data[i][0], zipped_data[i][1][0]);
    }
}

