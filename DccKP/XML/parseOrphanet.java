

import org.w3c.dom.*;
import javax.xml.parsers.*;
import java.io.File;
import java.util.*;

public class DisorderXMLParser {
    
    public static class DisorderInfo {
        public String id;
        public String orphaCode;
        public String expertLink;
        public String name;
        public List<AssociationInfo> associations;
        
        public DisorderInfo() {
            this.associations = new ArrayList<>();
        }
    }
    
    public static class AssociationInfo {
        public String targetId;
        public String targetOrphaCode;
        public String targetName;
        public String associationType;
    }
    
    public static Map<String, DisorderInfo> extractDisorderMap(String xmlFilePath) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new File(xmlFilePath));
        
        Map<String, DisorderInfo> disorderMap = new HashMap<>();
        
        NodeList disorderNodes = document.getElementsByTagName("Disorder");
        
        for (int i = 0; i < disorderNodes.getLength(); i++) {
            Element disorder = (Element) disorderNodes.item(i);
            String disorderId = disorder.getAttribute("id");
            
            DisorderInfo info = new DisorderInfo();
            info.id = disorderId;
            
            // Extract basic information
            info.orphaCode = getElementText(disorder, "OrphaCode");
            info.expertLink = getElementText(disorder, "ExpertLink");
            info.name = getElementTextByLang(disorder, "Name", "en");
            
            // Extract associations
            NodeList associations = disorder.getElementsByTagName("DisorderDisorderAssociation");
            for (int j = 0; j < associations.getLength(); j++) {
                Element association = (Element) associations.item(j);
                
                AssociationInfo assocInfo = new AssociationInfo();
                
                Element targetDisorder = (Element) association.getElementsByTagName("TargetDisorder").item(0);
                if (targetDisorder != null) {
                    assocInfo.targetId = targetDisorder.getAttribute("id");
                    assocInfo.targetOrphaCode = getElementText(targetDisorder, "OrphaCode");
                    assocInfo.targetName = getElementTextByLang(targetDisorder, "Name", "en");
                }
                
                Element assocType = (Element) association.getElementsByTagName("DisorderDisorderAssociationType").item(0);
                if (assocType != null) {
                    assocInfo.associationType = getElementTextByLang(assocType, "Name", "en");
                }
                
                info.associations.add(assocInfo);
            }
            
            disorderMap.put(disorderId, info);
        }
        
        return disorderMap;
    }
    
    private static String getElementText(Element parent, String tagName) {
        NodeList nodes = parent.getElementsByTagName(tagName);
        if (nodes.getLength() > 0) {
            return nodes.item(0).getTextContent();
        }
        return null;
    }
    
    private static String getElementTextByLang(Element parent, String tagName, String lang) {
        NodeList nodes = parent.getElementsByTagName(tagName);
        for (int i = 0; i < nodes.getLength(); i++) {
            Element element = (Element) nodes.item(i);
            if (lang.equals(element.getAttribute("lang"))) {
                return element.getTextContent();
            }
        }
        return null;
    }
    
    // Simple version - just ID to Name mapping
    public static Map<String, String> extractSimpleDisorderMap(String xmlFilePath) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new File(xmlFilePath));
        
        Map<String, String> simpleMap = new HashMap<>();
        
        NodeList disorderNodes = document.getElementsByTagName("Disorder");
        
        for (int i = 0; i < disorderNodes.getLength(); i++) {
            Element disorder = (Element) disorderNodes.item(i);
            String disorderId = disorder.getAttribute("id");
            String name = getElementTextByLang(disorder, "Name", "en");
            
            if (name != null) {
                simpleMap.put(disorderId, name);
            }
        }
        
        return simpleMap;
    }
    
    public static void main(String[] args) {
        try {
            Map<String, DisorderInfo> disorderMap = extractDisorderMap("your_file.xml");
            
            for (Map.Entry<String, DisorderInfo> entry : disorderMap.entrySet()) {
                DisorderInfo info = entry.getValue();
                System.out.println("Disorder ID: " + entry.getKey());
                System.out.println("  Name: " + info.name);
                System.out.println("  OrphaCode: " + info.orphaCode);
                System.out.println("  Associations: " + info.associations.size());
                for (AssociationInfo assoc : info.associations) {
                    System.out.println("    -> " + assoc.targetName + " (" + assoc.associationType + ")");
                }
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


