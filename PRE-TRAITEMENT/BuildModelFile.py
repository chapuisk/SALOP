import os
import numpy as np
import argparse

def removeEndLine(splittedLine: str) -> str:
    try:
        result = splittedLine.split(";")[0].split(":")[0]
        # Among parsing => Keep array form
        if "[" not in result:
            result = result.split(" ")[-2]

        if result == "":
            raise
    except:
        result = splittedLine.split(";")[0]

    return result

def extract_ExperimentLine(line: str) -> dict:
    result = {"name": "", "type": "INT", "value_min": "0", "value_max": "", "value_among": "",
              "varName": removeEndLine(line.split("var:")[1]), "condition": False}

    # Damien's particular variable
    #print(line)
    if result["varName"] == "force_parameters":
        result = None
    else:
        # Normal line
        try:
            result["name"] = line.split("\"")[1]

        except:
            result["name"] = line.split("\'")[1]
        try:
            result["value_min"] = removeEndLine(line.split("min:")[1])

            if (result["value_min"] == "true") or (result["value_min"] == "false") or( "true" in result["value_among"]) or( "false" in result["value_among"]):
                result["value_min"] = str(0)
                result["value_max"] = str(1)
                result["type"] = "BOOLEAN"

            else:
                if "among" not in line:
                    result["value_min"] = removeEndLine(line.split("min:")[1])
                    result["value_max"] = removeEndLine(line.split("max:")[1])
                else:
                    result["value_among"] = removeEndLine(line.split("among:")[1])
                    result["value_min"] = str(0)
                    result["value_max"] = str(1)

                if ('.' in result["value_min"]) or ('.' in result["value_max"]) :
                    result["type"] = "FLOAT"
        except:
            return None

    return result
def extractParametersAttributes(parameterLine: str ) -> dict:
    stringExtractor = parameterLine[0:2]
    result = None

    # It's an explicite parameter
    if stringExtractor == "pa":
        result = extract_ExperimentLine(parameterLine)
    # It's within a facet
    else:
        pass

    #check if the variable hasn't already be saved
    for p in parametersList:
        if result is not None and result["varName"] == p["varName"]:
            result = None
            break

    return result

def generateExperimentUniverse( gamlFilePath ,expName):
    right_experimentation = False
    in_commentary=False

    #Turn them all in absolute path
    gamlFilePath = os.path.abspath(gamlFilePath)


    # 1 _ Gather all parameters
    # #

    with open(gamlFilePath) as f:
        for l in f.readlines():
            if "/*" in l:
                in_commentary=True
            if "*/" in l:
                in_commentary=False
            if "method" in l:
                right_experimentation=False
            if "experiment" in l:
                right_experimentation=False
            if expName in l:
                right_experimentation=True
            if ("parameter" in l) and right_experimentation and (not in_commentary):
                temp = extractParametersAttributes( l.strip()  )
                if temp is not None:
                    parametersList.append( temp )

    return parametersList

def register_problem(problem,path):
    file= open(path, 'w')
    file.write(str(problem["num_vars"] )+";")
    file.write(str(problem["names"])+";")
    file.write(str(problem["bounds"]))
    file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s  -xml <experiment name> /path/to/file.gaml -o /path/to/model.txt')
    parser.add_argument('-xml', metavar=("<experiment name>", "/path/to/file.gaml"), nargs=2,
                        help='Classical xml arguments', required=True)
    parser.add_argument('-o',"--output",metavar="/path/to/model.txt", type =str ,help="Path to save model.txt",required=True)

    args = parser.parse_args()

    expName,gamlFilePath=args.xml

    path_model=args.output

    print("=== Reading .gaml file...\n")
    # 1 _ Gather all parameters
    #
    parametersList = []
    parametersList = generateExperimentUniverse(gamlFilePath, expName)

    print("Number of parameters: " + str(len(parametersList)))

    # 2 _ Generate all parameters value using saltelli sampling
    #
    print("=== Creating sampling...\n")

    parameters_name = []
    for parameter in parametersList:
        parameters_name.append(parameter["name"])

    bounds = np.zeros((len(parametersList), 2))
    i = 0
    for parameter in parametersList:
        bounds[i][0] = parameter["value_min"]
        bounds[i][1] = parameter["value_max"]
        i = i + 1

    #Creation of a problem to match with the saltelli method
    problem = {
        'num_vars': len(parametersList),
        'names': parameters_name,
        'bounds': bounds
    }
    register_problem(problem,path_model)