import { NextRequest, NextResponse } from "next/server";
import puppeteer from "puppeteer-core";
import chromium from "@sparticuz/chromium-min";

export const dynamic = "force-dynamic";
export const maxDuration = 60; // Allow up to 60 seconds for PDF generation

// Remote chromium binary URL for serverless environments
const CHROMIUM_REMOTE_URL = "https://github.com/Sparticuz/chromium/releases/download/v131.0.0/chromium-v131.0.0-pack.tar";

// Generate HTML report from data
function generateReportHTML(data: any): string {
  const {
    strokeType = "serve",
    grade = "B+",
    overallScore = 75,
    frames = 0,
    duration = 0,
    riskScore = 80,
    riskCounts = { high: 0, medium: 0, low: 0 },
    shoulderScore = 80,
    hipScore = 70,
    kneeScore = 75,
    elbowScore = 80,
    avgShoulder = null,
    minShoulder = null,
    maxShoulder = null,
    avgHip = null,
    avgKnee = null,
    avgElbow = null,
    llmResponse = "",
    deviationReport = null,
    generatedAt = new Date().toLocaleDateString(),
  } = data;

  // Convert markdown-like LLM response to HTML
  const formatLLMResponse = (text: string) => {
    if (!text) return "<p>No AI coaching feedback available.</p>";

    return text
      .replace(/## (.*?)$/gm, '<h2 class="section-title">$1</h2>')
      .replace(/### (.*?)$/gm, '<h3 class="subsection-title">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/^\- (.*?)$/gm, '<div class="list-item"><span class="bullet">•</span><span>$1</span></div>')
      .replace(/^(\d+)\. (.*?)$/gm, '<div class="numbered-item"><span class="number">$1.</span><span>$2</span></div>')
      .replace(/\n\n/g, "</p><p>");
  };

  // Helper for score colors
  const getScoreColor = (score: number) => {
    if (score >= 80) return '#10b981'; // green
    if (score >= 50) return '#f59e0b'; // amber
    return '#ef4444'; // red
  };

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StrikeSense Performance Report</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: #ffffff;
      color: #171717;
      line-height: 1.5;
      padding: 40px;
    }
    
    .container {
      max-width: 800px;
      margin: 0 auto;
    }
    
    .header {
      text-align: center;
      margin-bottom: 40px;
      padding-bottom: 20px;
      border-bottom: 2px solid #e5e5e5;
    }
    
    .logo {
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    
    .subtitle {
      color: #737373;
      font-size: 14px;
    }
    
    .report-title {
      font-size: 32px;
      font-weight: 700;
      margin: 20px 0 8px;
    }
    
    .report-meta {
      color: #737373;
      font-size: 14px;
    }
    
    .grade-section {
      background: #fafafa;
      border: 1px solid #e5e5e5;
      border-radius: 16px;
      padding: 32px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .grade-label {
      color: #737373;
      font-size: 14px;
      margin-bottom: 4px;
    }
    
    .grade-value {
      font-size: 64px;
      font-weight: 800;
    }
    
    .grade-percent {
      font-size: 28px;
      font-weight: 700;
      margin-left: 12px;
    }
    
    .color-green { color: #10b981; }
    .color-amber { color: #f59e0b; }
    .color-red { color: #ef4444; }
    .bg-green { background: #10b981; }
    .bg-amber { background: #f59e0b; }
    .bg-red { background: #ef4444; }
    
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 24px;
    }
    
    .metric-card {
      background: #fafafa;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 16px;
    }
    
    .metric-label {
      color: #737373;
      font-size: 11px;
      font-weight: 500;
      margin-bottom: 8px;
    }
    
    .metric-value {
      font-size: 24px;
      font-weight: 700;
      color: #000;
    }
    
    .metric-subtext {
      color: #a3a3a3;
      font-size: 11px;
      margin-top: 4px;
    }

    .metric-target {
      font-size: 10px;
      color: #a3a3a3;
      margin-top: 6px;
      padding-top: 6px;
      border-top: 1px solid #e5e5e5;
    }
    
    .metric-target span {
      color: #525252;
      font-weight: 500;
    }
    
    .section {
      background: #fafafa;
      border: 1px solid #e5e5e5;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
    }
    
    .section-header {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .progress-bar-container {
      margin-bottom: 16px;
    }
    
    .progress-label {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    
    .progress-name {
      font-size: 14px;
      color: #525252;
    }
    
    .progress-detail {
      font-size: 12px;
      color: #a3a3a3;
    }
    
    .progress-value {
      font-size: 14px;
      font-weight: 600;
      color: #000;
    }
    
    .progress-bar {
      height: 8px;
      background: #e5e5e5;
      border-radius: 4px;
      overflow: hidden;
    }
    
    .progress-fill {
      height: 100%;
      border-radius: 4px;
    }
    
    .list-item, .numbered-item {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 13px;
      color: #525252;
      margin-bottom: 8px;
      line-height: 1.6;
    }
    
    .list-item .bullet, .numbered-item .number {
      flex-shrink: 0;
      min-width: 16px;
      font-weight: 600;
    }
    
    .ai-feedback {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 20px;
    }
    
    .ai-feedback .section-title {
      font-size: 16px;
      font-weight: 600;
      margin: 20px 0 12px;
      color: #000;
    }
    
    .ai-feedback .section-title:first-child {
      margin-top: 0;
    }
    
    .ai-feedback .subsection-title {
      font-size: 14px;
      font-weight: 600;
      margin: 16px 0 8px;
      color: #525252;
    }
    
    .ai-feedback p {
      font-size: 13px;
      color: #525252;
      margin-bottom: 12px;
      line-height: 1.6;
    }
    
    .ai-feedback ul, .ai-feedback ol {
      margin: 12px 0;
      padding-left: 20px;
    }
    
    .ai-feedback li {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 13px;
      color: #525252;
      margin-bottom: 8px;
      line-height: 1.6;
    }
    
    .footer {
      text-align: center;
      padding-top: 24px;
      border-top: 1px solid #e5e5e5;
      color: #a3a3a3;
      font-size: 12px;
    }
    
    .page-break {
      page-break-before: always;
    }

    .deviation-card {
      background: #ffffff;
      border-left: 4px solid #f59e0b;
      padding: 16px;
      margin-bottom: 12px;
      border-radius: 8px;
      border: 1px solid #e5e5e5;
      border-left-width: 4px;
    }
    
    .deviation-card.critical {
      border-left-color: #ef4444;
    }
    
    .deviation-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    
    .deviation-title {
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .deviation-score {
      font-weight: 700;
      font-size: 16px;
    }
    
    .deviation-meta {
      display: flex;
      gap: 16px;
      font-size: 12px;
      color: #525252;
      margin-bottom: 8px;
    }
    
    .deviation-rec {
      background: #fafafa;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      color: #171717;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">🏓 StrikeSense</div>
      <div class="subtitle">AI-Powered Pickleball Analysis</div>
      <h1 class="report-title">Performance Report</h1>
      <p class="report-meta">${strokeType.charAt(0).toUpperCase() + strokeType.slice(1)} Analysis • ${generatedAt}</p>
    </div>
    
    <div class="grade-section">
      <div>
        <p class="grade-label">Overall Score</p>
        <div style="display: flex; align-items: baseline;">
          <span class="grade-value" style="color: ${getScoreColor(overallScore)};">${grade}</span>
          <span class="grade-percent" style="color: ${getScoreColor(overallScore)};">${overallScore}%</span>
        </div>
      </div>
      <div style="text-align: right;">
        <p class="grade-label">Analysis Summary</p>
        <p style="font-size: 14px; color: #525252;">${frames} frames • ${duration.toFixed(1)}s duration</p>
      </div>
    </div>
    
    <div class="metrics-grid">
      <div class="metric-card">
        <p class="metric-label">FRAMES ANALYZED</p>
        <p class="metric-value">${frames}</p>
        <p class="metric-subtext">${duration.toFixed(1)}s duration</p>
      </div>
      <div class="metric-card">
        <p class="metric-label">FORM SAFETY</p>
        <p class="metric-value" style="color: ${getScoreColor(riskScore)}">${riskScore}%</p>
        <p class="metric-subtext">${riskCounts.high > 0 ? `${riskCounts.high} caution` : "Looking good"}</p>
        <p class="metric-target">Target: <span>100%</span></p>
      </div>
      <div class="metric-card">
        <p class="metric-label">AVG HIP ROTATION</p>
        <p class="metric-value" style="color: ${avgHip !== null ? (avgHip >= 30 ? '#10b981' : avgHip >= 15 ? '#f59e0b' : '#ef4444') : '#000'}">${avgHip ? avgHip.toFixed(0) : "--"}</p>
        <p class="metric-subtext">degrees</p>
        <p class="metric-target">Target: <span>&gt;30°</span></p>
      </div>
      <div class="metric-card">
        <p class="metric-label">SHOULDER RANGE</p>
        <p class="metric-value" style="color: ${maxShoulder !== null ? (maxShoulder >= 90 ? '#10b981' : maxShoulder >= 60 ? '#f59e0b' : '#ef4444') : '#000'}">${minShoulder !== null ? minShoulder.toFixed(0) : "--"}-${maxShoulder !== null ? maxShoulder.toFixed(0) : "--"}°</p>
        <p class="metric-subtext">degrees</p>
        <p class="metric-target">Target: <span>&lt;120°</span></p>
      </div>
    </div>
    


    ${deviationReport && deviationReport.topDeviations.length > 0 ? `
    <div class="section">
      <h2 class="section-header">🎯 What to Improve</h2>
      <p style="font-size: 13px; color: #525252; margin-bottom: 20px;">${deviationReport.summary}</p>
      
      ${deviationReport.topDeviations.map((param: any, idx: number) => `
        <div class="deviation-card ${param.status === 'critical' ? 'critical' : ''}" style="border-left-color: ${param.status === 'critical' ? '#ef4444' : '#f59e0b'}">
          <div class="deviation-header">
            <div class="deviation-title">
              <span style="background: ${param.performanceImpact === 'high' ? '#ef4444' : '#f59e0b'}; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">${idx + 1}</span>
              <span>${param.label}</span>
              ${param.performanceImpact === 'high' ? '<span style="background: #fef2f2; color: #ef4444; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700;">HIGH IMPACT</span>' : ''}
            </div>
            <div class="deviation-score" style="color: ${getScoreColor(param.score)}">
              ${param.score}/100
            </div>
          </div>
          
          <div class="deviation-meta">
            <div><span style="color: #a3a3a3;">Your value:</span> <strong>${param.userValue}°</strong></div>
            <div><span style="color: #a3a3a3;">Optimal:</span> <strong style="color: #10b981;">${param.optimalRange.min}-${param.optimalRange.max}°</strong></div>
          </div>
          
          <div class="deviation-rec">
            💡 ${param.recommendation}
          </div>
        </div>
      `).join('')}
    </div>
    ` : ''}
    
    <div class="section">
      <h2 class="section-header">🤖 AI Coach Analysis</h2>
      <div class="ai-feedback">
        ${formatLLMResponse(llmResponse)}
      </div>
    </div>
    
    <div class="footer">
      <p>Generated by StrikeSense • ${generatedAt}</p>
      <p style="margin-top: 4px;">© 2024 StrikeSense • AI-Powered Pickleball Analysis</p>
    </div>
  </div>
</body>
</html>
  `;
}

export async function POST(request: NextRequest) {
  let browser = null;

  try {
    const body = await request.json();

    // Generate HTML from data
    const html = generateReportHTML(body);

    // Launch browser with @sparticuz/chromium-min using remote binary
    browser = await puppeteer.launch({
      args: chromium.args,
      defaultViewport: { width: 1200, height: 800 },
      executablePath: await chromium.executablePath(CHROMIUM_REMOTE_URL),
      headless: true,
    });

    const page = await browser.newPage();

    // Set content
    await page.setContent(html, {
      waitUntil: "networkidle0",
    });

    // Generate PDF
    const pdfBuffer = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: {
        top: "20px",
        right: "20px",
        bottom: "20px",
        left: "20px",
      },
    });

    await browser.close();
    browser = null;

    // Return PDF
    return new NextResponse(Buffer.from(pdfBuffer), {
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="strikesense-report-${Date.now()}.pdf"`,
      },
    });
  } catch (error: any) {
    console.error("PDF generation error:", error);
    return NextResponse.json(
      { error: "Failed to generate PDF", details: error.message },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

