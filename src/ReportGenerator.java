package src;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

public class ReportGenerator {

    // Generates a bar chart for model performance
    public static void generatePerformanceChart(Map<String, Double> modelMetrics, String outputPath) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        // Add metrics to the dataset
        for (Map.Entry<String, Double> entry : modelMetrics.entrySet()) {
            dataset.addValue(entry.getValue(), "F1 Score", entry.getKey());
        }

        // Create a bar chart
        JFreeChart chart = ChartFactory.createBarChart(
                "Model Performance Metrics",
                "Model",
                "F1 Score",
                dataset
        );

        // Save the chart as an image
        ChartUtils.saveChartAsPNG(new File(outputPath + "/performance_chart.png"), chart, 800, 600);
    }

    // Generates an HTML report summarizing results
    public static void generateHTMLReport(String outputPath, Map<String, Double> modelMetrics, String biasSummary) throws IOException {
        FileWriter writer = new FileWriter(outputPath + "/report.html");

        // Write the HTML structure
        writer.write("<!DOCTYPE html>");
        writer.write("<html>");
        writer.write("<head>");
        writer.write("<title>Bias Detection Report</title>");
        writer.write("</head>");
        writer.write("<body>");
        writer.write("<h1>AI Bias Detection Report</h1>");

        // Add performance metrics
        writer.write("<h2>Model Performance</h2>");
        writer.write("<table border='1'>");
        writer.write("<tr><th>Model</th><th>F1 Score</th></tr>");
        for (Map.Entry<String, Double> entry : modelMetrics.entrySet()) {
            writer.write("<tr><td>" + entry.getKey() + "</td><td>" + entry.getValue() + "</td></tr>");
        }
        writer.write("</table>");

        // Add bias summary
        writer.write("<h2>Bias Analysis</h2>");
        writer.write("<p>" + biasSummary + "</p>");

        // Add chart
        writer.write("<h2>Performance Chart</h2>");
        writer.write("<img src='performance_chart.png' alt='Performance Chart'>");

        writer.write("</body>");
        writer.write("</html>");
        writer.close();
    }

    public static void main(String[] args) {
        // Example data
        Map<String, Double> modelMetrics = Map.of(
                "ALBERT", 0.82,
                "TinyBERT", 0.79,
                "Logistic Regression", 0.75
        );

        String biasSummary = "The model shows minimal bias across gender and age groups. Demographic parity and equal opportunity metrics are within acceptable thresholds.";

        String outputPath = "output";

        // Ensure the output directory exists
        new File(outputPath).mkdirs();

        try {
            // Generate the performance chart
            generatePerformanceChart(modelMetrics, outputPath);

            // Generate the HTML report
            generateHTMLReport(outputPath, modelMetrics, biasSummary);

            System.out.println("Report generated successfully!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
