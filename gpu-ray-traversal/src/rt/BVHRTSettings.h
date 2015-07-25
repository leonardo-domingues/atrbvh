#pragma once

#include <string>
#include <vector>

/// <summary> This is a parser for a very simple configuration file to BVHRT. Its purpuse is to 
///           allow settings to quickly be changed from within the script that runs my experiments 
///           without having to recompile the code. It is ugly, but it gets the job done. 
/// </summary>
///
/// <remarks> Leonardo, 02/09/2015. </remarks>
class BVHRTSettings
{
public:    
    BVHRTSettings();
    ~BVHRTSettings();
    
    bool lbvh32;
    bool lbvh64;
    bool trbvh;
    bool atrbvh;
    bool collapseTree;
    int treeletSize;
    int iterations;

private:    
    void ParseConfigurationFile(char* fileLocation);
    bool Contains(std::vector<std::string> container, std::string token) const;
    template<class T> bool GetValue(std::vector<std::string> tokens, std::string id, T& value) 
            const;

};

