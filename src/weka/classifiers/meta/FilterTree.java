package weka.classifiers.meta;

import weka.classifiers.RandomizableClassifier;
import weka.core.*;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Random;

public class FilterTree extends RandomizableClassifier {

    //the code is contain 3 part: the tree, distribution, tostring

//    static final long serialVersionUID = 2767537273715121226L;

    //Option
    // The filter to use locally at each node
    protected Filter FilterTemplate = new AllFilter();

    // The minimum number of instances required for splitting
    protected double m_Threshold = 2.0;

    //The data that should be stored

    // The root node of the decision tree
    protected Node RootNode;

    // A random number generator
    protected Random random;


    @OptionMetadata(
            displayName = "threshold",
            description = "The minimum number of instances required for splitting (default = 2.0).",
            commandLineParamName = "M", commandLineParamSynopsis = "-M <double>",
            displayOrder = 1)
    public double getThreshold() {
        return m_Threshold;
    }

    public void setThreshold(double threshold) {
        this.m_Threshold = threshold;
    }

    @OptionMetadata(
            displayName = "filter",
            description = "The filter to use for splitting data, including filter options (default = AllFilter).",
            commandLineParamName = "F", commandLineParamSynopsis = "-F <filter specification>",
            displayOrder = 2)
    public Filter getFilter() {
        return FilterTemplate;
    }

    public void setFilter(Filter filter) {
        this.FilterTemplate = filter;
    }

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class for building a classification tree with local filter models for defining splits.";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }


    /**
     * An interface indicating objects storing node information, implemented by three node info classes.
     */
    protected interface NodeInfo extends Serializable {
    }

    /**
     * Class whose objects represent split nodes.
     */
    protected class SplitNodeInfo implements NodeInfo {

        // The attribute used for splitting
        protected Attribute SplitAttribute;
        protected Filter SplitFilter;
        // The split value
        protected double SplitValue;

        // The array of successor nodes
        protected Node Right;
        protected Node Left;

        /**
         * Constructs a SplitNodeInfo object
         *
         * @param splitAttribute the attribute that defines the split
         * @param splitValue     the value used for the split
         * @param left           the left node
         * @param right          the right node
         */
        public SplitNodeInfo(Attribute splitAttribute, double splitValue, Node left, Node right, Filter filter) {
            SplitAttribute = splitAttribute;
            SplitValue = splitValue;
            Left = left;
            Right = right;
            SplitFilter = filter;
        }
    }

    /**
     * Class whose objects represent leaf nodes.
     */
    protected class LeafNodeInfo implements NodeInfo {

        // The array of predictions
        protected double[] Prediction;

        /**
         * Constructs a LeafNodeInfo object.
         *
         * @param prediction the array of predictions to store at this node
         */
        public LeafNodeInfo(double[] prediction) {
            Prediction = prediction;
        }
    }

    /**
     * Class whose objects represent unexpanded nodes.
     * The rootNode
     */
    protected class UnexpandedNodeInfo implements NodeInfo {

        // The data to be used for expanding the node.
        protected Instances Data;

        /**
         * Constructs an UnexpandedNodeInfo object.
         *
         * @param data the data to be used for turning this node into an expanded node.
         */
        public UnexpandedNodeInfo(Instances data) {
            Data = data;
        }
    }

    /**
     * Class representing a node in the decision tree.
     */
    protected class Node implements Serializable {

        // The node information object that stores the actual information for this node.
        protected NodeInfo NodeInfo;

        /**
         * Constructs a node based on the give node info.
         *
         * @param nodeInfo an appropriate node information object
         */
        public Node(FilterTree.NodeInfo nodeInfo) {
            NodeInfo = nodeInfo;
        }
    }

    //build Classifier and distribution
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);
        //remove instances with missing class
        instances.deleteWithMissingClass();

        random = instances.getRandomNumberGenerator(getSeed());
        //set RootNode
        RootNode = new Node(new UnexpandedNodeInfo(instances));
        //building tree
        RootNode = makeTree(RootNode);

    }

    /**
     * Method for building the Filter tree.
     */
    protected Node makeTree(Node node) throws Exception {
        /*
         * filtering data.
         * Split: based on filtered data
         * Pass: based on unfiltered data
         */
        Instances unfilteredData = ((UnexpandedNodeInfo) node.NodeInfo).Data;
        if (Utils.smOrEq(unfilteredData.numInstances(), m_Threshold)) {
            return makeLeaf(node);
        }

        Filter origin_filter = getFilter();
        //make a deep copy of filter for use
        Filter filter = Filter.makeCopy(origin_filter);
        //use Random filter
        if (filter instanceof Randomizable) {
            ((Randomizable) filter).setSeed(random.nextInt());
        }

        filter.setInputFormat(unfilteredData);
            //original use type for filter
//        for (int i = 0; i < unfilteredData.numInstances(); i++) {
//            filter.input(unfilteredData.instance(i));
//        }
//        filter.batchFinished();
        //get the filtered data
//        Instances filteredData = filter.getOutputFormat();
//        Instance processed;
//        while ((processed = filter.output()) != null) {
//            filteredData.add(processed);
//        }

        //Use filter (more efficient)
        Instances filteredData = Filter.useFilter(unfilteredData,filter);

        // Compute attribute with maximum information gain.
        SplitInfo[] infoGainInfo = new SplitInfo[filteredData.numAttributes()];
        double[] infoGains = new double[filteredData.numAttributes()];
        Enumeration attEnum = filteredData.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGainInfo[att.index()] = computeInfoGain(filteredData, att);
            infoGains[att.index()] = infoGainInfo[att.index()].entropy;
        }
        //select the attribute
        int maxIndex = Utils.maxIndex(infoGains);
        Attribute m_Attribute = filteredData.attribute(maxIndex);

        // Create leaf if information gain is zero
        // Otherwise create the splitNode.
        if (Utils.smOrEq(infoGains[m_Attribute.index()], 0)) {
            return makeLeaf(node);
        } else {
            //Split the instance and then
            return makeSplitNode(node, m_Attribute, infoGainInfo[maxIndex], filteredData, filter);
        }

    }


    /**
     * Method that makes the given node into a leaf node by replacing the node information.
     *
     * @param node the node to turn into a leaf node
     * @return the leaf node
     */
    protected Node makeLeaf(Node node) {

        Instances data = ((UnexpandedNodeInfo) node.NodeInfo).Data;
        if (data.numInstances() == 0){
            return null;
        }
        double[] pred;
        if (data.classAttribute().isNumeric()) {
            double sum = 0;
            for (Instance instance : data) {
                sum += instance.classValue();
            }
            pred = new double[1];
            pred[0] = sum / (double) data.numInstances();
        } else {
            pred = new double[data.numClasses()];
            for (Instance instance : data) {
                pred[(int) instance.classValue()]++;
            }
            //normalize would change [n, m] to [1,0]
//            Utils.normalize(pred);
        }
        node.NodeInfo = new LeafNodeInfo(pred);
        return node;
    }

    /**
     * Method that makes the given node into a leaf node by replacing the node information.
     * It uses the recursive method to get the children tree or leaf (nodes).
     *
     * @param node the node to turn into a split node
     * @return the Split node
     */
    protected Node makeSplitNode(Node node, Attribute splitAttribute, SplitInfo splitInfo, Instances filteredData, Filter filter) throws Exception {

        Instances data = ((UnexpandedNodeInfo) node.NodeInfo).Data;

        Node[] childrenNode = new Node[2];
        double splitValue = splitInfo.splitValue;
        Instances[] subsets = new Instances[2];
        subsets[0] = new Instances(data, data.numInstances());
        subsets[1] = new Instances(data, data.numInstances());

        for (int i = 0; i < filteredData.numInstances(); i++) {
            if (filteredData.get(i).value(splitAttribute) < splitValue) {
                subsets[0].add(data.get(i));
            } else {
                subsets[1].add(data.get(i));
            }
        }

        childrenNode[0] = new Node(new UnexpandedNodeInfo(subsets[0]));
        childrenNode[1] = new Node(new UnexpandedNodeInfo(subsets[1]));

        node.NodeInfo = new SplitNodeInfo(splitAttribute, splitValue, makeTree(childrenNode[0]), makeTree(childrenNode[1]), filter);
        return node;
    }


    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att  the attribute
     * @return the information gain for the given attribute and data
     * @throws Exception if computation fails
     */
    private SplitInfo computeInfoGain(Instances data, Attribute att) throws Exception {

        double infoGain = computeEntropy(data);
        if (Utils.smOrEq(infoGain,0.0)){
            return new SplitInfo(0,0);
        }
//        SplitInfo splitInfo = computeSplitValue(data, att);
//        infoGain -= splitInfo.entropy;
//        splitInfo.entropy = infoGain;
        return computeSplitValue(data, att,infoGain);
    }


    //a class to store split info
    protected class SplitInfo {
        double splitValue;
        double entropy;

        public SplitInfo(double splitValueIn, double entropyIn) {
            splitValue = splitValueIn;
            entropy = entropyIn;
        }
    }

    /**
     * Computes the entropy of a dataset.
     *
     * @param unSortData the instance data for which entropy is to be computed
     * @return the value of Max entropy in the data's class distribution and the entropy
     * @throws Exception if computation fails
     */
//    private SplitInfo computeSplitValue(Instances unSortData, Attribute attribute) throws Exception {
//
//        //make a copy
//        Instances data = new Instances(unSortData);
//        //sort it
//        data.sort(attribute);
//
//        //storage info
//        double[] entropy = new double[data.numInstances()-1];
//        double splitValue;
//        double valuePoint;
//        int splitIndex;
//        int insCount = data.numInstances();
//        Instances[] subset = new Instances[2];
//
//        //initialization
//        subset[0] = new Instances(data, data.numInstances());
//        subset[1] = new Instances(data);
//        valuePoint = data.get(0).value(attribute);
//        //here is the index problem
//        for (int i = 0; i < insCount -1; i++) {
//            subset[0].add(data.get(i));
//            subset[1].remove(0);
//            entropy[i] = (computeEntropy(subset[0]) * i/insCount)
//                    + (computeEntropy(subset[1]) * (insCount - i)/insCount);
//        }
//        splitIndex = Utils.minIndex(entropy);
//        //use this and next data to get the approx data. Not work if many value is the same
//        splitValue = (data.get(splitIndex).value(attribute) + data.get(splitIndex + 1).value(attribute))/2;
////        splitValue = data.get(splitIndex).value(attribute) ;
//        //return the final result
//        SplitInfo splitInfo = new SplitInfo(splitValue, entropy[splitIndex]);
//
//        return splitInfo;
//    }
    /**
     * Computes the entropy of a dataset.
     *
     * @param unSortData the instance data for which entropy is to be computed
     * @return the value of Max infoGain in the data's class distribution and the infoGain
     * @throws Exception if computation fails
     */
    private SplitInfo computeSplitValue(Instances unSortData, Attribute attribute, double originalEntropy) throws Exception {
        //make a copy
        Instances data = new Instances(unSortData);
        //sort it
        data.sort(attribute);

        //storage info
        double[] entropy = new double[data.numInstances()-1];
        double splitValue;
        double valuePoint;
        double currentAttributeValue;
        double[] left;
        double[] right;
        int splitIndex;
        int insCount = data.numInstances();
        int currentClassValue;


        //initialization
        valuePoint = data.get(0).value(attribute);
        right = classList(data);
        left = new double[right.length];

        //first time, must run!!!
      //here is very risky!
        currentClassValue = (int)data.get(0).classValue();
        left[currentClassValue] ++;
        right[currentClassValue]--;

        //here is the index problem
        for (int i = 1; i < insCount ; i++) {
            currentClassValue = (int)data.get(i).classValue();
            currentAttributeValue = data.get(i).value(attribute);
            //if the current value is not change, set the infoGain to 0
            if (currentAttributeValue == valuePoint){
                left[currentClassValue] ++;
                right[currentClassValue]--;
                entropy[i-1] = 0.0;
            }else {
                // compute the entropy
                valuePoint = currentAttributeValue;
                entropy[i-1] = originalEntropy
                        - computeEntropy(left) * (i+1)/ insCount
                        - computeEntropy(right) * (insCount-i-1)/ insCount;
                //then update the calss
                left[currentClassValue] ++;
                right[currentClassValue]--;
            }
        }

        splitIndex = Utils.maxIndex(entropy);
        //check the split point
        if (splitIndex == 0){
            splitValue = (data.get(splitIndex).value(attribute) + data.get(splitIndex + 1).value(attribute))/2;
        }

        else{
            splitValue = (data.get(splitIndex).value(attribute) + data.get(splitIndex + 1).value(attribute))/2;
        }


        //return the final result
        SplitInfo splitInfo = new SplitInfo(splitValue, entropy[splitIndex]);

        return splitInfo;
    }



    /**
     * Computes the entropy of a dataset.
     *
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data's class distribution
     * @throws Exception if computation fails
     */
    private double computeEntropy(Instances data) throws Exception {

        double[] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }


    /**
     * Computes the entropy of a dataset.
     *
     * @param classCounts the classCounts[] for which entropy is to be computed
     * @return the entropy of the data's class distribution
     * @throws Exception if computation fails
     */
    private double computeEntropy(double[] classCounts) throws Exception {

        double entropy = 0;
        for (int j = 0; j < classCounts.length; j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        double totalNum = 0;
        for (double num:classCounts) {
            totalNum += num;
        }
        entropy /= totalNum;
        return entropy + Utils.log2(totalNum);
    }

    /**
     * Get the class List
     *
     * @param data the data input to get the class
     * @return classCounts the double[] that contains the class data
     * @throws Exception if processing fails
     */
    private double[] classList(Instances data) throws Exception {

        double[] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        return classCounts;
    }

    /**
     * Method that updates the given estimates based on the given instance and the subtree attached to the given node.
     *
     * @param distribution the estimates to be updated
     * @param instance     the instance for which estimates are to be updated
     * @param node         the node whose subtree we are considering
     */
    protected void distributionForInstance(double[] distribution, Instance instance, Node node) throws Exception {

        if (node.NodeInfo instanceof LeafNodeInfo) {
            //if it is the leaf, jusKt check th prediction
            for (int i = 0; i < distribution.length; i++) {
                distribution[i] += ((LeafNodeInfo) node.NodeInfo).Prediction[i];
            }
//            distribution = ((LeafNodeInfo) node.NodeInfo).Prediction;
        } else {
            //if it is the split point, useF nodeFilter to filter the instance
            SplitNodeInfo splitInfo = (SplitNodeInfo) node.NodeInfo;
            Filter filter = ((SplitNodeInfo) node.NodeInfo).SplitFilter;
            filter.input(instance);
            Instance filteredInstance = filter.output();
            //check the split value to get the direction to next node
            if (filteredInstance.value(splitInfo.SplitAttribute.index()) < splitInfo.SplitValue)
            {
                distributionForInstance(distribution, instance, splitInfo.Left);
            }else {
                distributionForInstance(distribution, instance, splitInfo.Right);
            }
        }

    }

    /**
     * Method that returns estimated class probabilities for the given instance if the class is nominal. If the
     * class is numeric, it will return a single-element array with the estimated target value.
     *
     * @param instance the instance for which a prediction is to be generated.
     * @return the estimates obtained from the tree
     */
    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] distribution = new double[instance.numClasses()];

        distributionForInstance(distribution, instance, RootNode);

        if (instance.classAttribute().isNominal()) {
            Utils.normalize(distribution);
        }

        return distribution;
    }


    /**
     * Method that updates the given estimates based on the given instance and the subtree attached to the given node.
     *
     * @param distribution the estimates to be updated
     * @param instances     the instances for which estimates are to be updated
     */
    protected void distributionForInstance(double[][] distribution, Instances instances) throws Exception {

            //if it is the split point, use nodeFilter to filter the instance
            SplitNodeInfo splitInfo = (SplitNodeInfo) RootNode.NodeInfo;
            Filter filter = ((SplitNodeInfo) RootNode.NodeInfo).SplitFilter;
            Instances filteredInstances = Filter.useFilter(instances,filter);

            for (int i = 0; i < filteredInstances.numInstances() ; i++) {
                if (filteredInstances.instance(i).value(splitInfo.SplitAttribute.index()) < splitInfo.SplitValue)
                {
                    distributionForInstance(distribution[i], instances.instance(i), splitInfo.Left);
                }else {
                    distributionForInstance(distribution[i], instances.instance(i), splitInfo.Right);
                }
            }
    }
    /**
     * Method that returns estimated class probabilities for the given instance if the class is nominal. If the
     * class is numeric, it will return a single-element array with the estimated target value.
     * This way should be more efficient
     *
     * @param instances the instances for which a prediction are to be generated.
     * @return the estimates obtained from the tree
     */
    public double[][] distributionForInstance(Instances instances) throws Exception {

        double[][] distribution = new double[instances.numInstances()][instances.numClasses()];

        distributionForInstance(distribution, instances);

        if (instances.classAttribute().isNominal()) {
            for (double[] pre : distribution) {
                Utils.normalize(pre);
            }
        }

        return distribution;
    }


        //set it to false when use the allFilter to compare
//    public boolean implementsMoreEfficientBatchPrediction() {
//        return true;
//    }
    /**
     * Method that returns a textual description of the subtree attached to the given node. The description is
     * returned in a string buffer.
     *
     * @param stringBuffer buffer to hold the description
     * @param node the node whose subtree is to be described
     * @param levelString the level of the node in the overall tree structure
     */
    protected void toString(StringBuffer stringBuffer, Node node, String levelString) {

        if (node.NodeInfo instanceof SplitNodeInfo) {
            stringBuffer.append("\n").append(levelString).append(((SplitNodeInfo) node.NodeInfo).SplitAttribute.name()).append(" < ").append(Utils.doubleToString(((SplitNodeInfo) node.NodeInfo).SplitValue, getNumDecimalPlaces()));
            toString(stringBuffer, ((SplitNodeInfo) node.NodeInfo).Left, levelString + "|   ");
            stringBuffer.append("\n").append(levelString).append(((SplitNodeInfo) node.NodeInfo).SplitAttribute.name()).append(" >= ").append(Utils.doubleToString(((SplitNodeInfo) node.NodeInfo).SplitValue, getNumDecimalPlaces()));
            toString(stringBuffer, ((SplitNodeInfo) node.NodeInfo).Right, levelString + "|   ");
        } else {
            double[] dist = ((LeafNodeInfo) node.NodeInfo).Prediction;
            stringBuffer.append(":");
            for (double pred : dist) {
                stringBuffer.append(" ").append(Utils.doubleToString(pred, getNumDecimalPlaces()));
            }
        }
    }

    /**
     * Method that returns a textual description of the classifier.
     *
     * @return the textual description as a string
     */
    public String toString() {

        if (RootNode == null) {
            return "FilterTree: No classifier built yet.";
        }
        StringBuffer stringBuffer = new StringBuffer();
        toString(stringBuffer, RootNode, "");
        return stringBuffer.toString();
    }


    /**
     * Main method to run this classifier from the command-line with the standard option handling.
     *
     * @param args the command-line options
     */
    public static void main(String[] args) {

        runClassifier(new FilterTree(), args);
    }

}
