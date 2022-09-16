import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

/**
 * This class create into a CSV file an OFATx1 or OFATx2 sampling
 */
public class OFAT {
    protected List<Map<String, Object>> MySample;
    protected Map<String, Map<String, List<Object>>> MyInputs;
    protected Map<String, Double> AnalyseValue;
    protected Map<String, Double> DefaultValue;
    protected List<String> ParametersNames;
    protected double seuil;
    protected Map<String,Double> moy;
    protected Map<String,Double> min;
    protected Map<String,Double> max;
    protected Map<String,Double> median;

    /**
     * Builder for OFATx1 and OFATx2
     * @param myInputs : List of Inputs with name, type and bounds
     * @param seuil : threshold for sensitivity analysis
     * @param AnalyseValue : List of results of sensitivity analysis
     * @param DefaultValue : Default value for each Inputs
     */
    public OFAT(Map<String, Map<String, List<Object>>> myInputs, double seuil, Map<String, Double> AnalyseValue, Map<String, Double> DefaultValue) {
        MyInputs = myInputs;
        ParametersNames = MyInputs.keySet().stream().toList();
        this.seuil = seuil;
        this.AnalyseValue = AnalyseValue;
        this.DefaultValue = DefaultValue;
        moy=new LinkedHashMap<>();
        min= new LinkedHashMap<>();
        max= new LinkedHashMap<>();
        median= new LinkedHashMap<>();
        if (AnalyseValue.size() != MyInputs.size()) {
            System.out.println("Not the same size: ");
            System.out.println(MyInputs.size());
            throw new RuntimeException();
        }
        ParametersNames.forEach(s -> {
            if (!AnalyseValue.containsKey(s)) {
                //Not the same parameters
                System.out.println(s + " not in AnalyseValue");
                throw new RuntimeException();
            }
        });
    }

    public Map<String,Object> addForAll(String name,Map<String,Object> sampleMap){
        for(int i=0;i<ParametersNames.size();i++){
            String tmpName=ParametersNames.get(i);
            if(name!=tmpName){
                sampleMap.put(tmpName,DefaultValue.get(tmpName));
            }
        }
        return sampleMap;

    }

    public Map<String,Object> addForAll2(List<String> names,Map<String,Object> sampleMap){
        for(int i=0;i<ParametersNames.size();i++){
            String tmpName=ParametersNames.get(i);
            if(!names.contains(tmpName)){
                sampleMap.put(tmpName,DefaultValue.get(tmpName));
            }
        }
        return sampleMap;

    }

    /**
     * Main method for OFATx1 sampling
     * @param precision: the number of value for each parameters
     * @return return the sampling
     */
    public List<Map<String, Object>> OFATSampling(int precision) {
        List<Map<String,Object>> samples= new ArrayList<>();
        for (int i = 0; i < AnalyseValue.size(); i++) {
            String name= ParametersNames.get(i);
            double tempVal = AnalyseValue.get(name);
            if (tempVal >= seuil) {
                if(MyInputs.get(name).containsKey("Int")){
                    int min= (int) MyInputs.get(name).get("Int").get(0);
                    int max= (int) MyInputs.get(name).get("Int").get(1);
                    double factor= 1/((double)precision);
                    for(int y=0;y<precision+1;y++){
                        double value =(double) Math.round(min+((factor*y)*(max-min)));
                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                        tmpMap=addForAll(name,tmpMap);
                        tmpMap.put(name,value);
                        samples.add((tmpMap));
                    }
                }else if(MyInputs.get(name).containsKey("Boolean")){
                    Map<String,Object> tmpMap= new LinkedHashMap<>();
                    tmpMap=addForAll(name,tmpMap);
                    tmpMap.put(name,true);
                    samples.add(tmpMap);
                    Map<String,Object> tmpMap2= new LinkedHashMap<>();
                    tmpMap2=addForAll(name,tmpMap2);
                    tmpMap2.put(name,false);
                    samples.add(tmpMap2);
                }else if(MyInputs.get(name).containsKey("Discret")){
                    List<Object> tmpList= MyInputs.get(name).get("Discret");
                    tmpList.forEach(el->{
                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                        tmpMap=addForAll(name,tmpMap);
                        tmpMap.put(name,el);
                        samples.add((tmpMap));
                    });
                }else if(MyInputs.get(name).containsKey("Double")){
                    double min= (double) MyInputs.get(name).get("Double").get(0);
                    double max= (double) MyInputs.get(name).get("Double").get(1);
                    double factor= 1/((double)precision);
                    for(int y=0;y<precision+1;y++){
                        double value =min+((factor*y)*(max-min));
                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                        tmpMap.put(name,value);
                        tmpMap=addForAll(name,tmpMap);
                        samples.add((tmpMap));
                    }
                }else{
                    System.out.println("Type not defined");
                }
            } else {
                System.out.print(ParametersNames.get(i));
                System.out.println(" Under the threshold");
            }
        }
        MySample=samples;
        return samples;
    }

    /**
     * Main method for OFATx2 sampling
     * @param precision
     * @return return the sampling
     */
    public List<Map<String, Object>> OFATSampling_x2(int precision) {
        List<Map<String,Object>> samples= new ArrayList<>();
        for(int i=0;i<AnalyseValue.size()-1;i++){
            String name= ParametersNames.get(i);
            double tempVal = AnalyseValue.get(name);
            if (tempVal >= seuil) {
                for(int y=i+1;y<AnalyseValue.size();y++){
                    String name_2= ParametersNames.get(y);
                    double tempVal_2 = AnalyseValue.get(name_2);
                    if (tempVal_2 >= seuil) {
                        if(MyInputs.get(name).containsKey("Int")) {
                            int min= (int) MyInputs.get(name).get("Int").get(0);
                            int max= (int) MyInputs.get(name).get("Int").get(1);
                            double factor= 1/((double)precision);
                            if(MyInputs.get(name_2).containsKey("Int")){
                                int min_2= (int) MyInputs.get(name_2).get("Int").get(0);
                                int max_2= (int) MyInputs.get(name_2).get("Int").get(1);
                                for(int z=0;z<precision+1;z++){
                                    double value =(double) Math.round(min+((factor*z)*(max-min)));
                                    for(int x=0;x<precision+1;x++){
                                        double value_2 =(double) Math.round(min_2+((factor*x)*(max_2-min_2)));
                                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                                        List<String> names= new ArrayList<>();
                                        names.add(name);
                                        names.add(name_2);
                                        tmpMap=addForAll2(names,tmpMap);
                                        tmpMap.put(name,value);
                                        tmpMap.put(name_2,value_2);
                                        samples.add((tmpMap));
                                    }
                                }
                            }else if(MyInputs.get(name_2).containsKey("Boolean")){
                                List<String> names= new ArrayList<>();
                                names.add(name);
                                names.add(name_2);
                                for(int z=0;z<precision+1;z++){
                                    double value =(double) Math.round(min+((factor*z)*(max-min)));
                                    Map<String,Object> tmpMap= new LinkedHashMap<>();
                                    tmpMap=addForAll2(names,tmpMap);
                                    tmpMap.put(name_2,true);
                                    tmpMap.put(name,value);
                                    samples.add(tmpMap);
                                    Map<String,Object> tmpMap2= new LinkedHashMap<>();
                                    tmpMap2=addForAll2(names,tmpMap2);
                                    tmpMap2.put(name_2,false);
                                    tmpMap2.put(name,value);
                                    samples.add(tmpMap2);
                                }
                            }else if(MyInputs.get(name_2).containsKey("Discret")){
                                System.out.println("Not do yet");
                            }else if(MyInputs.get(name_2).containsKey("Double")){
                                double min_2= (double) MyInputs.get(name_2).get("Double").get(0);
                                double max_2= (double) MyInputs.get(name_2).get("Double").get(1);
                                List<String> names= new ArrayList<>();
                                names.add(name);
                                names.add(name_2);
                                for(int z=0;z<precision+1;z++){
                                    double value =(double) Math.round(min+((factor*z)*(max-min)));
                                    for(int x=0;x<precision+1;x++){
                                        double value_2 =min_2+((factor*x)*(max_2-min_2));
                                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                                        tmpMap=addForAll2(names,tmpMap);
                                        tmpMap.put(name,value);
                                        tmpMap.put(name_2,value_2);
                                        samples.add((tmpMap));
                                    }
                                }
                            }else{
                                System.out.println("Type not defined");
                            }
                        }else if(MyInputs.get(name).containsKey("Boolean")){
                            System.out.println(" Boolean method, not done");
                        }else if(MyInputs.get(name).containsKey("Discret")){
                            System.out.println("discrete method, not done");
                        }else if(MyInputs.get(name).containsKey("Double")){
                            double min= (double) MyInputs.get(name).get("Double").get(0);
                            double max= (double) MyInputs.get(name).get("Double").get(1);
                            double factor= 1/((double)precision);
                            if(MyInputs.get(name_2).containsKey("Int")){
                                int min_2= (int) MyInputs.get(name_2).get("Int").get(0);
                                int max_2= (int) MyInputs.get(name_2).get("Int").get(1);
                                for(int z=0;z<precision+1;z++){
                                    double value =min+((factor*z)*(max-min));
                                    for(int x=0;x<precision+1;x++){
                                        double value_2 =(double) Math.round(min_2+((factor*x)*(max_2-min_2)));
                                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                                        List<String> names= new ArrayList<>();
                                        names.add(name);
                                        names.add(name_2);
                                        tmpMap=addForAll2(names,tmpMap);
                                        tmpMap.put(name,value);
                                        tmpMap.put(name_2,value_2);
                                        samples.add((tmpMap));
                                    }
                                }
                            }else if(MyInputs.get(name_2).containsKey("Boolean")){
                                List<String> names= new ArrayList<>();
                                names.add(name);
                                names.add(name_2);
                                for(int z=0;z<precision+1;z++){
                                    double value =min+((factor*z)*(max-min));
                                    Map<String,Object> tmpMap= new LinkedHashMap<>();
                                    tmpMap=addForAll2(names,tmpMap);
                                    tmpMap.put(name_2,true);
                                    tmpMap.put(name,value);
                                    samples.add(tmpMap);
                                    Map<String,Object> tmpMap2= new LinkedHashMap<>();
                                    tmpMap2=addForAll2(names,tmpMap2);
                                    tmpMap2.put(name_2,false);
                                    tmpMap2.put(name,value);
                                    samples.add(tmpMap2);
                                }
                            }else if(MyInputs.get(name_2).containsKey("Discret")){
                                System.out.println("Not do yet");
                            }else if(MyInputs.get(name_2).containsKey("Double")){
                                double min_2= (double) MyInputs.get(name_2).get("Double").get(0);
                                double max_2= (double) MyInputs.get(name_2).get("Double").get(1);
                                List<String> names= new ArrayList<>();
                                names.add(name);
                                names.add(name_2);
                                for(int z=0;z<precision+1;z++){
                                    double value =min+((factor*z)*(max-min));
                                    for(int x=0;x<precision+1;x++){
                                        double value_2 =min_2+((factor*x)*(max_2-min_2));
                                        Map<String,Object> tmpMap= new LinkedHashMap<>();
                                        tmpMap=addForAll2(names,tmpMap);
                                        tmpMap.put(name,value);
                                        tmpMap.put(name_2,value_2);
                                        samples.add((tmpMap));
                                    }
                                }
                            }else{
                                System.out.println("Type not defined");
                            }
                        }else{
                            System.out.println("Type not defined");
                        }
                    }
                }
            }

        }
        MySample=samples;
        return samples;
    }

    /**
     * Method to save the sampling in a CSV file
     * @param file
     * @throws Exception
     */
    public void saveSimulation(File file) throws Exception {
        try{
            FileWriter fw = new FileWriter(file, false);
            fw.write(this.buildSimulationCsv());
            fw.close();
        }catch (Exception e) {
            throw new Exception();
        }
    }

    private String buildSimulationCsv() {
        StringBuffer sb= new StringBuffer();
        String separator=",";
        String ln="\n";
        List<String> varNames= MySample.get(0).keySet().stream().toList();
        varNames.forEach(name-> sb.append(name).append(separator));
        sb.delete(sb.length()-1,sb.length());
        sb.append(ln);
        for (Map<String, Object> VarMap : MySample) {
            IntStream.range(0, varNames.size()).forEach(i -> {
                sb.append(VarMap.get(varNames.get(i)));
                sb.append(separator);
            });
            sb.delete(sb.length() - 1, sb.length());
            sb.append(ln);
        }
        return sb.toString();
    }

    public static void main(String[] args) throws Exception {

    }
}
