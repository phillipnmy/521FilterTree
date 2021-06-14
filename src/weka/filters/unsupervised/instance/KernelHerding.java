package weka.filters.unsupervised.instance;

import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.filters.SimpleBatchFilter;

import java.util.Random;

public class KernelHerding extends SimpleBatchFilter {

    /** for serialization */
    static final long serialVersionUID = -251831442047263433L;

    /** The kernel function to use. */
    protected Kernel m_Kernel = new PolyKernel();

    /** The subsample size, percent of original set, default 100% */
    protected double m_SampleSizePercent = 100;

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = getKernel().getCapabilities();
        result.setOwner(this);

        result.setMinimumNumberInstances(0);

        result.enableAllClasses();
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        return result;
    }

    /** Handling the kernel parameter. */
    @OptionMetadata(
            displayName = "Kernel function",
            description = "The kernel function to use.", displayOrder = 1,
            commandLineParamName = "K",
            commandLineParamSynopsis = "-K <kernel specification>")
    public Kernel getKernel() {  return m_Kernel; }
    public void setKernel(Kernel value) { m_Kernel = value; }

    /** Handling the parameter setting the sample size. */
    @OptionMetadata(
            displayName = "Percentage of the training set to sample.",
            description = "The percentage of the training set to sample (between 0 and 100).", displayOrder = 3,
            commandLineParamName = "Z",
            commandLineParamSynopsis = "-Z <double>")
    public void setSampleSizePercent(double newSampleSizePercent) { m_SampleSizePercent = newSampleSizePercent; }
    public double getSampleSizePercent() { return m_SampleSizePercent; }

    @Override
    public String globalInfo() { return "A filter implementing kernel herding for unsupervised subsampling of data."; }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return new Instances(inputFormat, 0);
    }

    protected Instances collected;
    protected Instance seed;
    @Override
    protected Instances process(Instances instances) throws Exception {

        // We only modify the first batch of data that this filter receives
        // (e.g., the training data of a classifier, not the test data)
        if (!isFirstBatchDone()) {

//            int seedIndex = 0;
//            seed = instances.get(0);
            //initialize the seed(the x1)
            int seedIndex = new Random().nextInt(instances.numInstances() - 1);
            seed = instances.get(seedIndex);
            //get output percentage num of instances
            double percent = getSampleSizePercent() / 100 ;
            //initialize the out put Instances
            this.collected = new Instances(instances,0);
            //num of input instances
            int total = instances.numInstances();
            //get the output num of instances
            int InsNum = (int) (total *  percent) + 1;
//            int InsNum = instances.numInstances();

            this.collected.add(seed);

            //if use random seed
            //int MaxInstanceIndex = seedIndex;
            //initial kernel
            this.m_Kernel.clean();
            this.m_Kernel.buildKernel(instances);

            //storage array for k(x,y), k(x, xt)
            double[] kxy = new double[total];
            for(int i = 0; i < total; i++)
            {
                for (int j = 0; j < total; j++){
                    kxy[i] += m_Kernel.eval(i, j, seed);
                }
                kxy[i] /= Double.valueOf(total);
            }
            //initial P2 related parameter

            int[] Index_xt = new int[InsNum];
            Index_xt[0] = seedIndex;
            double[] SUM_XT = new double[total];

            for(int f = 1; f < InsNum; f++){//loop from 1 to T
                //Initial max value
                double maxValue = Double.MIN_NORMAL;// MIN_VALUE0x1.0p-1022
                int MaxInstanceIndex = seedIndex;
                for ( int i = 0; i < total; i++){//loop every x in X
                    SUM_XT[i] += P2(Index_xt[f-1],i);
                    double k = kxy[i] - SUM_XT[i]/Double.valueOf(f +1.0);
                        if (maxValue <= k){
                            maxValue = k;
                            MaxInstanceIndex = i;
                            Index_xt[f] = i ;
                        }
                }

                //add to collect
                this.collected.add(instances.get(MaxInstanceIndex));
            }

            this.m_Kernel.clean();
            //signal for done
            this.m_FirstBatchDone = true;
        //System.out.println("what");
            instances = new Instances(this.collected);
            //return instances
        }

        //return this.collected;
        return instances;
    }
    private double P2(int InsNum, int Index) throws Exception {
        double res = m_Kernel.eval(Index, InsNum, seed);
        return res;
    }

    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runFilter(new KernelHerding(), options);
    }
}
