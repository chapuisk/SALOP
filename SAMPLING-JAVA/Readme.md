# Readme for SAMPLING JAVA
## Context
in this Readme, you will find all information you need for the use of "OFAT.java"

This java class can create sampling for OFATx1 and OFATx2
## Use
You have to right in the main (or use another class) something like:  
```
        Map<String,Map<String,List<Object>>> inputs= new LinkedHashMap();
        
        //For Infection probability
        List<Object> l1=new ArrayList<>();l1.add(0.01);
        l1.add(1.0);
        Map<String,List<Object>> type1= new HashMap<>();
        type1.put("Double",l1);
        inputs.put("infection_probability",type1);
        
        //For Dodge Probability
        List<Object> l2=new ArrayList<>();
        l2.add(0.0);
        l2.add(0.9);
        Map<String,List<Object>> type2= new HashMap<>();
        type2.put("Double",l2);
        inputs.put("proba_dodge_disease",type2);
        
        //For Number of initial infected
        List<Object> l3=new ArrayList<>();
        l3.add(1);
        l3.add(2146);
        Map<String,List<Object>> type3= new HashMap<>();
        type3.put("Int",l3);
        inputs.put("nb_infected_init",type3);
        
        //For Cure Probability
        List<Object> l4=new ArrayList<>();
        l4.add(0.001);
        l4.add(0.1);
        Map<String,List<Object>> type4= new HashMap<>();
        type4.put("Double",l4);
        inputs.put("proba_to_cure",type4);
        
        Map<String,Double> analyse=new LinkedHashMap<>();
        analyse.put("infection_probability",10.0);
        analyse.put("proba_dodge_disease",10.0);
        analyse.put("nb_infected_init",10.0);
        analyse.put("proba_to_cure",9.0);
        
        Map<String,Double> defaultv = new LinkedHashMap<>();
        defaultv.put("infection_probability",0.05);
        defaultv.put("proba_dodge_disease",0.025);
        defaultv.put("nb_infected_init",5.0);
        defaultv.put("proba_to_cure",0.001);
        
        double threshold= 7;
        // Number of value to visit for each Input
        int precision = 10;
        OFAT o= new OFAT(inputs,threshold,analyse,defaultv);
        List<Map<String, Object>> sample= o.OFATSampling_x2(precision);
        o.OFATSampling_x2(20);
        File f=new File("./OFATx2.csv");
        o.saveSimulation(f);
```