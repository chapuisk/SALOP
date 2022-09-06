import argparse
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from SALib.sample import saltelli
import SALib.sample.morris

"""
This is the Generator XML file for Sensitivity Analysis. This file will generate XML according with the number of samples required and the number of machines to use.
This script work with different files and script.
This script use some relative path that match with the project folder. If you want to use this script outside of this folder, you have to change some path.
This is a list of some relative path to change with explanation:
 - In func: register_problem(problem):
     file= open('../SA/model_problem_analysis.txt', 'w') -> This write the model in a file, use to know for the other script what is the model (Parameters names and bounds)
"""

def autoIndexSelector( argsFName : str ) -> int :
	path, tail = os.path.split( argsFName )

	return len([f for f in os.listdir(path)
		if (tail[:-4] in f) and os.path.isfile(os.path.join(path, f))])

def createXmlFiles(experimentName: str, gamlFilePath: str, saltelli_values, parametersList: list, xmlFilePath: str, replication: int = 1, split: int = -1, output: str = "../../batch_output", seed: int = 0, final: int = -1, until: str = "") -> bool:
    xmlFilePath = os.path.abspath(xmlFilePath)

    # Prevent wrong path
    if output[-1] == "/":
        output = output[:-1]

    # Create output
    os.makedirs(os.path.dirname(xmlFilePath), exist_ok=True)

    # Create XML
    root = ET.Element("Experiment_plan")
    new_allParamValues = []
    hasCondition = False
    xmlNumber = autoIndexSelector(xmlFilePath)
    localSeed = seed

    #Generate the good path
    tmp= xmlFilePath.split("/.*/")
    new_str=""
    for i in range(0,len(tmp)):
        new_str=new_str+"../"

    # Number of replication for every simulation
    for i in range(replication):
        # Every dot in the explorable universe
        for k in range(len(saltelli_values[parametersList[0]["name"]])):
            #print(k)
            resultSubFolder = ""

            simu = ET.SubElement(root, "Simulation", {
                "id"	: str( localSeed - seed ),
                "seed"		: str( localSeed ),
                "experiment": experimentName,
                "sourcePath": new_str+gamlFilePath
            })
            if final != -1:
                simu.set("finalStep", str(final))
            if until != "":
                simu.set("until", str(until))

            parameters = ET.SubElement(simu, "Parameters")
            # Set values for every parameter in the experiment
            for j in range(len(parametersList)):
                # Set exploration point
                canWriteParameter = False
                if not parametersList[j]["condition"]:
                    canWriteParameter = True
                else:
                    condition = parametersList[j]["condition"].split(",")
                    for p in parameters:
                        if p.get("var") == condition[0] and p.get("value") == condition[1]:
                            canWriteParameter = hasCondition = True

                if canWriteParameter:
                    ET.SubElement(parameters, "Parameter", {
                        "name"	: parametersList[j]["name"],
                        "type"	: parametersList[j]["type"],
                        "value" : str(int(saltelli_values[parametersList[j]["name"]][k])) if parametersList[j]["type"] == "INT" else str(saltelli_values[parametersList[j]["name"]][k]),
                        "var"	: parametersList[j]["varName"]
                    })
                    resultSubFolder += parametersList[j]["varName"] + "_" + str(saltelli_values[parametersList[j]["name"]][k]) + "-"

            # On first round, prevent duplicated simulation (caused by condition)
            # + create new universe space list (without this duplication)
            duplicate = False
            if i == 0 and hasCondition:

                for sim in reversed(root[:-1]):
                    if ET.tostring(parameters, encoding='unicode').replace("</Parameters>", "") in ET.tostring(sim, encoding='unicode') :
                        duplicate = True
                        break

                if duplicate:
                    # Remove duplicated element from XML
                    root.remove(root[-1])
                else:
                    # Create cleared parameter list
                    tmp_values = []
                    for m in range(0,len(parametersList)-1):
                        tmp_values.append(saltelli_values[parametersList[m]["name"]][k])
                    new_allParamValues.append(tmp_values)
            ET.SubElement(simu, "Outputs")

            # Write and flush XML root if have to split
            if len(list(root)) >= split != -1:
                tree = ET.ElementTree(root)
                tree.write(xmlFilePath[:-4 ] +"-" +str(xmlNumber ) +".xml")

                root = ET.Element("Experiment_plan")
                xmlNumber = xmlNumber + 1

            # Prepare for next loop
            localSeed = localSeed +1

        # Reset universe space list without duplicated simulations
        if i == 0 and len(new_allParamValues) > 0 and hasCondition:
            allParamValues = new_allParamValues

    # File write of the (last?) XML file
    tree = ET.ElementTree(root)
    if xmlNumber == 0:
        tree.write(xmlFilePath)
    elif len(root) > 0:
        tree.write(xmlFilePath[:-4] + "-" + str(xmlNumber) + ".xml")

    return True


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

def register_problem(problem):
    file= open('./model_problem_analysis.txt', 'w')
    file.write(str(problem["num_vars"] )+";")
    file.write(str(problem["names"])+";")
    file.write(str(problem["bounds"]))
    file.close()

if __name__ == '__main__':
    # 0 _ Get/Set parameters
    #
    parser = argparse.ArgumentParser(usage='$ python3 %(prog)s [options] -f INT -xml <experiment name> /path/to/file.gaml /path/to/file.xml')
    parser.add_argument('-r', '--replication', metavar='INT',
                        help="Number of replication for each parameter space (default: 1)", default=1, type=int)
    parser.add_argument('-s', '--split', metavar='INT', help="Number of machines (or nodes for HPC) (default: 1)",
                        default=1, type=int)
    parser.add_argument('-S', '--seed', metavar='INT', help="Starting value for seeding simulation (default: 0)",
                        default=0, type=int)
    parser.add_argument('-f', '--final', metavar='INT', help="Final step for simulations", default=-1, type=int,
                        required=True)
    parser.add_argument('-xml', metavar=("<experiment name>", "/path/to/file.gaml", "/path/to/file.xml"), nargs=3,
                        help='Classical xml arguments', required=True)
    parser.add_argument('-sample', metavar='INT',help="Number of sampling (default:128)", default=128,type=int)
    parser.add_argument('-analysis',metavar='STR',help="Method analysis (default:sobol), can be: sobol/morris", default="sobol",type=str)
    args = parser.parse_args()

    # 0 _ Set used variables
    #
    expName, gamlFilePath, xmlFilePath = args.xml
    nb_sample=args.sample
    type_analysis=args.analysis
    parametersList= []

    print("=== Reading .gaml file...\n")
    # 1 _ Gather all parameters
    #
    parametersList = generateExperimentUniverse(gamlFilePath,expName)

    print("Number of parameters: "+str(len(parametersList)))

    # 2 _ Generate all parameters value using saltelli sampling
    #
    print("=== Creating sampling...\n")


    parameters_name=[]
    for parameter in parametersList:
        parameters_name.append(parameter["name"])

    bounds=np.zeros((len(parametersList), 2))
    i=0
    for parameter in parametersList:
        bounds[i][0]=parameter["value_min"]
        bounds[i][1] = parameter["value_max"]
        i=i+1

    #Creation of a problem to match with the saltelli method
    problem = {
        'num_vars': len(parametersList),
        'names': parameters_name,
        'bounds': bounds
    }
    register_problem(problem)

    param_values=None
    if type_analysis=="sobol":
        #Saltelli sampling
        print("=== Sobol sampling starting...\n")
        param_values = saltelli.sample(problem, nb_sample)
        #print(param_values)
    else:
        if type_analysis=="morris":
            print("=== Morris sampling starting...\n")
            param_values = SALib.sample.morris.sample(problem, nb_sample, num_levels=4)


    saltelli_values={}
    i=0
    for parameter in parametersList:
        if parameter["value_among"] !="":
            if parameter["type"]=="BOOLEAN":
                tmp_values=[]
                for value in param_values[:,i]:
                    if value >=0.5:
                        tmp_values.append(1)
                    else:
                        tmp_values.append(0)
            else:
                tmp_values=[]
                stringtemp=parameter["value_among"]
                stringtemp=re.sub("\[","",stringtemp)
                stringtemp=re.sub("\]", "",stringtemp)
                values_among_tmp=re.split(",",stringtemp)
                nb_value=len(values_among_tmp)
                for value in param_values[:,i]:
                    if 0 <= value < 1/nb_value:
                        tmp_values.append(float(values_among_tmp[0]))
                    else:
                        if 1 >= value > (1 / (nb_value))*(nb_value - 1):
                            tmp_values.append(float(values_among_tmp[nb_value-1]))
                    for y in range(1,nb_value-1):
                        if 1/nb_value*y <= value < 1/(nb_value)*(y + 1):
                            tmp_values.append(float(values_among_tmp[y]))
                    else:
                        tmp_values.append(0)

            saltelli_values.update({parameter["name"]:np.array(tmp_values)})
        else:
            if parameter["type"]=="BOOLEAN":
                tmp_values=[]
                for value in param_values[:,i]:
                    if value >=0.5:
                        tmp_values.append(1)
                    else:
                        tmp_values.append(0)
                saltelli_values.update({parameter["name"]:np.array(tmp_values)})
            else:
                if parameter["type"]=="INT":
                    tmp_values = []
                    for value in param_values[:,i]:
                        tmp_values.append(round(value))
                    saltelli_values.update({parameter["name"]: np.array(tmp_values)})
                else:
                    saltelli_values.update({parameter["name"]: param_values[:,i]})
        i = i + 1

    print("=== Sampling done\n")
    # 3 _ Generate XML
    #
    number_of_machine= args.split
    number_for_split= round(len(saltelli_values[parametersList[0]["name"]])/number_of_machine)


    print("=== Start generating XML files...\n")
    if createXmlFiles(expName, gamlFilePath, saltelli_values, parametersList, xmlFilePath, args.replication, number_for_split,"../../batch_output",args.seed, args.final):
        print("\n=== Done ")
        print("=== END ")
    else:
        print("\n=== Error :(")
