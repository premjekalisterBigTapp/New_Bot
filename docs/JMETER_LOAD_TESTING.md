# JMeter Load Testing Guide for BigTapp Agentic Chatbot

**Production URL:** `https://chatbot.bigtapp.com`  
**Protocol:** HTTPS (Port 443)  
**Last Updated:** January 2026

> ⚠️ **Important:** The production server **enforces HTTPS**. HTTP requests are automatically redirected to HTTPS via 301 redirect, which breaks POST requests. Always use HTTPS.

This document provides **comprehensive, step-by-step instructions** for load testing the BigTapp Agentic Chatbot using Apache JMeter.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [JMeter Installation & Setup](#2-jmeter-installation--setup)
3. [Creating a New Test Plan](#3-creating-a-new-test-plan)
4. [Configuring HTTP Request Defaults](#4-configuring-http-request-defaults)
5. [Adding Thread Group (Virtual Users)](#5-adding-thread-group-virtual-users)
6. [Creating HTTP Samplers](#6-creating-http-samplers)
7. [Adding CSV Data Set for Test Messages](#7-adding-csv-data-set-for-test-messages)
8. [Configuring Assertions](#8-configuring-assertions)
9. [Adding Listeners (Results & Reports)](#9-adding-listeners-results--reports)
10. [Running the Load Test](#10-running-the-load-test)
11. [Analyzing Results](#11-analyzing-results)
12. [API Reference](#12-api-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### 1.1 Software Requirements

| Software | Version | Download Link |
|----------|---------|---------------|
| Java JDK | 8 or higher | [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://adoptium.net/) |
| Apache JMeter | 5.5 or higher | [JMeter Download](https://jmeter.apache.org/download_jmeter.cgi) |

### 1.2 Verify Java Installation

Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

```bash
java -version
```

**Expected Output:**
```
java version "17.0.x" 2024-xx-xx LTS
Java(TM) SE Runtime Environment (build 17.0.x+x-LTS-xxx)
```

If Java is not installed, download and install it before proceeding.

### 1.3 Network Requirements

- Ensure your machine can reach `https://chatbot.bigtapp.com` on **port 443**
- Test connectivity:
  ```bash
  curl -I https://chatbot.bigtapp.com/health
  ```
  **Expected:** `HTTP/1.1 200 OK`

---

## 2. JMeter Installation & Setup

### 2.1 Installing JMeter

1. **Download JMeter:**
   - Go to https://jmeter.apache.org/download_jmeter.cgi
   - Download the **Binaries** → `apache-jmeter-5.x.zip` (or `.tgz` for Linux/Mac)

2. **Extract the Archive:**
   - Windows: Right-click → `Extract All...` → Choose destination folder (e.g., `C:\Tools\apache-jmeter-5.6.3`)
   - Linux/Mac: `tar -xzf apache-jmeter-5.x.tgz`

3. **Launch JMeter:**
   - Windows: Navigate to `bin` folder → Double-click `jmeter.bat`
   - Linux/Mac: Navigate to `bin` folder → Run `./jmeter.sh`

### 2.2 JMeter Memory Configuration (Recommended)

For load testing, increase JMeter's heap size:

1. Navigate to `<JMETER_HOME>/bin/`
2. Edit `jmeter.bat` (Windows) or `jmeter.sh` (Linux/Mac)
3. Find and modify the heap settings:

**Windows (`jmeter.bat`):**
```batch
set HEAP=-Xms1g -Xmx4g -XX:MaxMetaspaceSize=256m
```

**Linux/Mac (`jmeter.sh`):**
```bash
HEAP="-Xms1g -Xmx4g -XX:MaxMetaspaceSize=256m"
```

---

## 3. Creating a New Test Plan

### 3.1 Create Test Plan

1. **Launch JMeter** (if not already open)
2. You will see a default **Test Plan** in the left panel
3. **Right-click on "Test Plan"** → Select `Rename`
4. Name it: `BigTapp Chatbot Load Test`

### 3.2 Configure Test Plan Properties

1. **Click on "BigTapp Chatbot Load Test"** in the left panel
2. In the right panel, configure:

| Property | Value | Explanation |
|----------|-------|-------------|
| Name | `BigTapp Chatbot Load Test` | Descriptive name for your test |
| Comments | `Load test for BigTapp Agentic Chatbot - HTTP` | Optional description |
| User Defined Variables | (leave empty for now) | We'll add variables later |
| Run Thread Groups consecutively | ☐ Unchecked | Allow parallel execution |
| Functional Test Mode | ☐ Unchecked | Keep unchecked for performance tests |

### 3.3 Save the Test Plan

1. Press `Ctrl + S` (or `Cmd + S` on Mac)
2. Save as: `BigTapp_Chatbot_LoadTest.jmx`
3. Choose a memorable location (e.g., `C:\JMeterTests\`)

---

## 4. Configuring HTTP Request Defaults

This sets default values for ALL HTTP requests in your test plan.

### 4.1 Add HTTP Request Defaults

1. **Right-click on "BigTapp Chatbot Load Test"** (the Test Plan)
2. Navigate: `Add` → `Config Element` → `HTTP Request Defaults`

### 4.2 Configure HTTP Request Defaults

Click on the newly added **"HTTP Request Defaults"** and configure:

#### Basic Tab Settings

| Field | Value | Description |
|-------|-------|-------------|
| **Protocol [http]** | `https` | ⚠️ **IMPORTANT: Must use HTTPS (server redirects HTTP)** |
| **Server Name or IP** | `chatbot.bigtapp.com` | The target server hostname |
| **Port Number** | `443` | Standard HTTPS port |
| **Path** | (leave empty) | We'll set paths per request |
| **Content Encoding** | `UTF-8` | Character encoding for requests |

#### Timeouts Section

| Field | Value (milliseconds) | Description |
|-------|---------------------|-------------|
| **Connect Timeout** | `60000` | 60 seconds to establish connection |
| **Response Timeout** | `120000` | 120 seconds for response (LLM can be slow) |

### 4.3 Verification

Your HTTP Request Defaults should look like this:

```
Protocol: https
Server Name: chatbot.bigtapp.com
Port: 443
Content Encoding: UTF-8
Connect Timeout: 60000
Response Timeout: 120000
```

---

## 5. Adding Thread Group (Virtual Users)

The Thread Group defines the number of virtual users (threads), ramp-up period, and test duration.

### 5.1 Add Thread Group

1. **Right-click on "BigTapp Chatbot Load Test"** (the Test Plan)
2. Navigate: `Add` → `Threads (Users)` → `Thread Group`

### 5.2 Configure Thread Group

Click on the newly added **"Thread Group"** and configure:

| Field | Value | Description |
|-------|-------|-------------|
| **Name** | `Chatbot Users` | Descriptive name for this group |
| **Comments** | `Virtual users sending chat messages` | Optional description |
| **Action to be taken after a Sampler error** | `Continue` | Don't stop on errors |

#### Thread Properties

| Property | Recommended Value | Description |
|----------|-------------------|-------------|
| **Number of Threads (users)** | `20` | Start with 20 concurrent users |
| **Ramp-Up Period (seconds)** | `30` | Time to start all 20 users (1 user/1.5 sec) |
| **Loop Count** | `10` | Each user sends 10 requests |

#### Alternative: Duration-Based Test

If you prefer time-based testing instead of loop count:

1. Check ☑️ **"Specify Thread lifetime"**
2. Configure:

| Property | Value | Description |
|----------|-------|-------------|
| **Duration (seconds)** | `300` | Run test for 5 minutes |
| **Startup delay (seconds)** | `0` | Start immediately |

### 5.3 Test Phases Recommendation

| Phase | Threads | Ramp-Up | Loop Count | Purpose |
|-------|---------|---------|------------|---------|
| **Smoke Test** | 5 | 10 sec | 5 | Verify test setup works |
| **Load Test** | 20 | 30 sec | 10 | Normal load simulation |
| **Stress Test** | 50 | 60 sec | 20 | Find breaking point |
| **Soak Test** | 20 | 20 sec | ∞ (Duration: 3600 sec) | Endurance test |

---

## 6. Creating HTTP Samplers

We'll create two types of requests: WhatsApp Webhook and Direct Chat API.

### 6.1 Add HTTP Header Manager (Required)

Before adding samplers, add a header manager:

1. **Right-click on "Chatbot Users"** (Thread Group)
2. Navigate: `Add` → `Config Element` → `HTTP Header Manager`
3. Click **"Add"** button at the bottom
4. Add this header:

| Name | Value |
|------|-------|
| `Content-Type` | `application/json` |

### 6.2 Create Direct Chat API Sampler (Recommended for Load Testing)

1. **Right-click on "Chatbot Users"** (Thread Group)
2. Navigate: `Add` → `Sampler` → `HTTP Request`
3. **Rename it:** `Chat API Request`

#### Configure the HTTP Request:

**Basic Tab:**

| Field | Value |
|-------|-------|
| **Name** | `Chat API Request` |
| **Comments** | `Direct agent chat API endpoint` |
| **Protocol [http]** | (leave empty - uses default) |
| **Server Name or IP** | (leave empty - uses default) |
| **Port Number** | (leave empty - uses default) |
| **Method** | `POST` |
| **Path** | `/agent-chat` |
| **Content Encoding** | `UTF-8` |

**Body Data Tab:**

1. Select the **"Body Data"** sub-tab
2. Enter the following JSON:

```json
{
  "session_id": "jmeter_${__threadNum}_${__UUID}",
  "message": "${message}"
}
```

> **Note:** `${__threadNum}` creates unique session per thread, `${__UUID}` ensures global uniqueness, and `${message}` will come from a CSV file.

### 6.3 Create WhatsApp Webhook Sampler (Optional - For Production Simulation)

1. **Right-click on "Chatbot Users"** (Thread Group)
2. Navigate: `Add` → `Sampler` → `HTTP Request`
3. **Rename it:** `WhatsApp Webhook`

#### Configure the HTTP Request:

**Basic Tab:**

| Field | Value |
|-------|-------|
| **Name** | `WhatsApp Webhook` |
| **Method** | `POST` |
| **Path** | `/webhook/whatsapp` |

**Body Data Tab:**

```json
{
  "object": "whatsapp_business_account",
  "entry": [
    {
      "id": "123456789",
      "changes": [
        {
          "value": {
            "messaging_product": "whatsapp",
            "metadata": {
              "display_phone_number": "6512345678",
              "phone_number_id": "788312047688093"
            },
            "contacts": [
              {
                "profile": {
                  "name": "JMeter User ${__threadNum}"
                },
                "wa_id": "659${__Random(1000000,9999999)}"
              }
            ],
            "messages": [
              {
                "from": "659${__Random(1000000,9999999)}",
                "id": "wamid.jmeter_${__UUID}",
                "timestamp": "${__time(/1000)}",
                "text": {
                  "body": "${message}"
                },
                "type": "text"
              }
            ]
          },
          "field": "messages"
        }
      ]
    }
  ]
}
```

### 6.4 Create Health Check Sampler (For Baseline)

1. **Right-click on "Chatbot Users"** (Thread Group)
2. Navigate: `Add` → `Sampler` → `HTTP Request`
3. **Rename it:** `Health Check`

**Configure:**

| Field | Value |
|-------|-------|
| **Name** | `Health Check` |
| **Method** | `GET` |
| **Path** | `/ready` |

---

## 7. Adding CSV Data Set for Test Messages

This allows you to use different messages for each request.

### 7.1 Create the CSV File

1. Open a text editor (Notepad, VS Code, etc.)
2. Create a file named `test_messages.csv`
3. Add the following content:

```csv
message
Hi, I need help with insurance
What travel insurance plans do you have?
Can you recommend a plan for a family trip?
How much does travel insurance cost?
What does the plan cover?
I want to claim for my medical bills
How do I submit a claim?
What is the coverage for trip cancellation?
Do you have annual travel plans?
What is the maximum coverage amount?
I am traveling to Japan next month
I need insurance for my family of 4
What pre-existing conditions are covered?
How long does claim processing take?
Can I extend my policy?
What documents do I need for a claim?
Is COVID-19 covered?
I want to cancel my policy
How do I renew my insurance?
What happens if my flight is delayed?
```

4. **Save the file** in your JMeter test directory (e.g., `C:\JMeterTests\test_messages.csv`)

### 7.2 Add CSV Data Set Config

1. **Right-click on "Chatbot Users"** (Thread Group)
2. Navigate: `Add` → `Config Element` → `CSV Data Set Config`

### 7.3 Configure CSV Data Set Config

| Field | Value | Description |
|-------|-------|-------------|
| **Name** | `Test Messages` | Descriptive name |
| **Filename** | `C:\JMeterTests\test_messages.csv` | **Full absolute path** to your CSV file |
| **File Encoding** | `UTF-8` | Character encoding |
| **Variable Names** | `message` | Column header name (matches CSV) |
| **Ignore first line** | `True` | Skip the header row |
| **Delimiter** | `,` | CSV delimiter |
| **Allow quoted data?** | `True` | Handle quoted strings |
| **Recycle on EOF?** | `True` | Restart from beginning when file ends |
| **Stop thread on EOF?** | `False` | Continue testing |
| **Sharing mode** | `All threads` | Share data across all threads |

---

## 8. Configuring Assertions

Assertions validate that responses are correct.

### 8.1 Add Response Assertion (For Chat API)

1. **Right-click on "Chat API Request"** (the HTTP Sampler)
2. Navigate: `Add` → `Assertions` → `Response Assertion`

**Configure:**

| Field | Value |
|-------|-------|
| **Name** | `Verify 200 OK` |
| **Apply to** | `Main sample only` |
| **Field to Test** | `Response Code` |
| **Pattern Matching Rules** | ☑️ `Equals` |
| **Patterns to Test** | `200` (click "Add" and type `200`) |

### 8.2 Add Response Assertion (For WhatsApp Webhook)

1. **Right-click on "WhatsApp Webhook"** (the HTTP Sampler)
2. Navigate: `Add` → `Assertions` → `Response Assertion`

**Configure:**

| Field | Value |
|-------|-------|
| **Name** | `Verify OK Response` |
| **Apply to** | `Main sample only` |
| **Field to Test** | `Text Response` |
| **Pattern Matching Rules** | ☑️ `Contains` |
| **Patterns to Test** | `OK` |

### 8.3 Add Duration Assertion

> ⚠️ **IMPORTANT:** Duration Assertion is a **different assertion type** from Response Assertion. Don't confuse them!

**Step-by-Step:**

1. **Right-click on "Chatbot_user"** (your Thread Group in the left panel)
2. Navigate to: `Add` → `Assertions` → `Duration Assertion`
   
   ```
   Right-click Thread Group → Add → Assertions → Duration Assertion
                                                  ↑
                                    (This is a SEPARATE menu item,
                                     NOT the same as Response Assertion)
   ```

3. The Duration Assertion dialog will open - it has a **very simple interface** with just one field

**Configure the Duration Assertion:**

| Field | Value |
|-------|-------|
| **Name** | `Duration Assertion` |
| **Duration in milliseconds** | `30000` |

That's it! The Duration Assertion only has one setting - the maximum allowed response time in milliseconds.

**What each assertion type does:**
- **Response Assertion** → Checks response content/code (text, status code, etc.)
- **Duration Assertion** → Checks if response time exceeds a threshold

> **Note:** LLM responses typically take 5-15 seconds. The 30000ms (30 seconds) threshold is a reasonable SLA for chatbot responses under load.

---

## 9. Adding Listeners (Results & Reports)

Listeners collect and display test results.

### 9.1 Add View Results Tree (For Debugging)

1. **Right-click on "BigTapp Chatbot Load Test"** (Test Plan)
2. Navigate: `Add` → `Listener` → `View Results Tree`

> ⚠️ **Warning:** Disable this during actual load tests as it consumes memory.

### 9.2 Add Summary Report

1. **Right-click on "BigTapp Chatbot Load Test"** (Test Plan)
2. Navigate: `Add` → `Listener` → `Summary Report`

**Understanding Summary Report Columns:**

| Column | Description |
|--------|-------------|
| **Label** | Name of the sampler |
| **# Samples** | Total requests sent |
| **Average** | Average response time (ms) |
| **Min** | Minimum response time (ms) |
| **Max** | Maximum response time (ms) |
| **Std. Dev.** | Standard deviation |
| **Error %** | Percentage of failed requests |
| **Throughput** | Requests per second |
| **Received KB/sec** | Data received rate |
| **Sent KB/sec** | Data sent rate |
| **Avg. Bytes** | Average response size |

### 9.3 Add Aggregate Report

1. **Right-click on "BigTapp Chatbot Load Test"** (Test Plan)
2. Navigate: `Add` → `Listener` → `Aggregate Report`

### 9.4 Add Response Time Graph

1. **Right-click on "BigTapp Chatbot Load Test"** (Test Plan)
2. Navigate: `Add` → `Listener` → `Response Time Graph`

### 9.5 Save Results to File (Optional)

For each listener, you can configure it to save results:

1. Click on the listener
2. In the **"Write results to file / Read from file"** field
3. Enter a path like: `C:\JMeterTests\results\summary_${__time(yyyyMMdd_HHmmss)}.csv`

---

## 10. Running the Load Test

### 10.1 Pre-Flight Checklist

Before running the test, verify:

- [ ] HTTP Request Defaults configured correctly (HTTP, port 80)
- [ ] Thread Group settings are appropriate for your phase
- [ ] CSV file path is correct and file exists
- [ ] Test plan is saved
- [ ] Target server is accessible (`curl http://chatbot.bigtapp.com/health`)

### 10.2 Run from GUI (For Debugging Only)

1. Click the **green "Start" button** (▶️) in the toolbar
2. Or press `Ctrl + R`
3. Monitor progress in listeners

> ⚠️ **Important:** GUI mode is for debugging only. Use CLI mode for actual load tests.

### 10.3 Run from Command Line (Recommended for Load Tests)

Open Command Prompt/Terminal and navigate to JMeter's `bin` directory:

**Windows:**
```batch
cd C:\Tools\apache-jmeter-5.6.3\bin
jmeter -n -t "C:\JMeterTests\BigTapp_Chatbot_LoadTest.jmx" -l "C:\JMeterTests\results\results.jtl" -e -o "C:\JMeterTests\results\html_report"
```

**Linux/Mac:**
```bash
cd /opt/apache-jmeter-5.6.3/bin
./jmeter -n -t "/home/user/JMeterTests/BigTapp_Chatbot_LoadTest.jmx" -l "/home/user/JMeterTests/results/results.jtl" -e -o "/home/user/JMeterTests/results/html_report"
```

**Command Explanation:**

| Flag | Description |
|------|-------------|
| `-n` | Non-GUI mode |
| `-t` | Path to test plan file (.jmx) |
| `-l` | Path to save results log (.jtl) |
| `-e` | Generate HTML report after test |
| `-o` | Output directory for HTML report |

### 10.4 Monitor Test Progress

During CLI execution, JMeter shows:
```
Starting standalone test @ 2026 Jan 19 10:15:00 IST (1737264300000)
Waiting for possible Shutdown/StopTestNow/HeapDump/ThreadDump message on port 4445
summary +    100 in 00:00:30 =    3.3/s Avg:  5234 Min:  1234 Max: 12456 Err:     0 (0.00%)
summary +    150 in 00:00:30 =    5.0/s Avg:  4521 Min:  1123 Max: 11234 Err:     2 (1.33%)
summary =    250 in 00:01:00 =    4.2/s Avg:  4805 Min:  1123 Max: 12456 Err:     2 (0.80%)
```

### 10.5 Stop the Test

**GUI Mode:** Click the red "Stop" button (⏹️) or press `Ctrl + .`

**CLI Mode:** Press `Ctrl + C` or send SIGTERM

---

## 11. Analyzing Results

### 11.1 Open HTML Report

After CLI mode test with `-e -o` flags, open:
```
C:\JMeterTests\results\html_report\index.html
```

### 11.2 Key Metrics to Analyze

| Metric | Target Value | Description |
|--------|--------------|-------------|
| **Error Rate** | < 1% | Percentage of failed requests |
| **Average Response Time** | < 10,000 ms | Mean response time |
| **90th Percentile** | < 15,000 ms | 90% of requests complete within this time |
| **99th Percentile** | < 25,000 ms | 99% of requests complete within this time |
| **Throughput** | > 5 req/sec | Requests processed per second |

### 11.3 Interpreting Results

**Good Performance:**
- Error rate < 1%
- Response times consistent (low std. deviation)
- Throughput stable throughout test

**Signs of Problems:**
- Error rate increasing over time → Memory leak or resource exhaustion
- Response times increasing → Server overload
- Throughput decreasing → Bottleneck reached

---

## 12. API Reference

### 12.1 WhatsApp Webhook Endpoint

| Property | Value |
|----------|-------|
| **URL** | `https://chatbot.bigtapp.com/webhook/whatsapp` |
| **Method** | `POST` |
| **Content-Type** | `application/json` |
| **Expected Response Code** | `200` |
| **Expected Response Body** | `OK` |

**Full Request Example:**

```bash
curl -X POST "https://chatbot.bigtapp.com/webhook/whatsapp" \
  -H "Content-Type: application/json" \
  -d '{
    "object": "whatsapp_business_account",
    "entry": [{
      "id": "123456789",
      "changes": [{
        "value": {
          "messaging_product": "whatsapp",
          "metadata": {
            "display_phone_number": "6512345678",
            "phone_number_id": "788312047688093"
          },
          "contacts": [{
            "profile": {"name": "Test User"},
            "wa_id": "6598765432"
          }],
          "messages": [{
            "from": "6598765432",
            "id": "wamid.test123",
            "timestamp": "1737264300",
            "text": {"body": "Hello, I need travel insurance"},
            "type": "text"
          }]
        },
        "field": "messages"
      }]
    }]
  }'
```

### 12.2 Agent Chat Endpoint

| Property | Value |
|----------|-------|
| **URL** | `https://chatbot.bigtapp.com/agent-chat` |
| **Method** | `POST` |
| **Content-Type** | `application/json` |
| **Expected Response Code** | `200` |

**Request Body Schema:**

```json
{
  "session_id": "string (required)",
  "message": "string (required)"
}
```

**Full Request Example:**

```bash
curl -X POST "https://chatbot.bigtapp.com/agent-chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_001",
    "message": "I want to buy travel insurance for Japan"
  }'
```

**Response Example:**

```json
{
  "response": "I'd be happy to help you with travel insurance for Japan! To recommend the best plan, could you please tell me...",
  "sources": "products/travel/asia_plan.pdf",
  "debug_state": {
    "product": "Travel",
    "intent": "rec",
    "slots_filled": ["destination"]
  }
}
```

### 12.3 Health Check Endpoints

| Endpoint | Method | Description | Expected Response |
|----------|--------|-------------|-------------------|
| `https://chatbot.bigtapp.com/health` | GET | Basic health check | `{"status": "ok"}` |
| `https://chatbot.bigtapp.com/ready` | GET | Readiness (checks Redis, Mongo, LLM) | `{"status": "ready", "services": {...}}` |
| `https://chatbot.bigtapp.com/metrics` | GET | Prometheus metrics | Text format metrics |

---

## 13. Troubleshooting

### 13.1 Common Errors

#### Error: Connection Refused

**Symptom:** `java.net.ConnectException: Connection refused`

**Solutions:**
1. Verify server is running: `curl https://chatbot.bigtapp.com/health`
2. Check firewall settings
3. Verify you're using HTTPS (port 443)

#### Error: 502 Bad Gateway

**Symptom:** Response code `502`

**Solutions:**
1. Backend server may be overloaded - reduce thread count
2. Check server logs for errors
3. Wait and retry

#### Error: 301 Moved Permanently + 405 Method Not Allowed

**Symptom:** 
- Response shows `301 Moved Permanently` with `location: https://...`
- Followed by `405 Method Not Allowed`
- POST requests appear as GET in the results

**Cause:** The server redirects HTTP to HTTPS. When JMeter follows the redirect, it converts POST to GET and loses the request body.

**Solutions:**
1. ⚠️ **Use HTTPS instead of HTTP** - Update HTTP Request Defaults:
   - Protocol: `https`
   - Port: `443`
2. This is the **only reliable solution** - the server enforces HTTPS

#### Error: 429 Too Many Requests

**Symptom:** Response code `429`

**Solutions:**
1. Rate limiting is active (10 requests/minute/session by default)
2. Use unique session IDs with `${__UUID}`
3. Add **Constant Timer** between requests:
   - Right-click on Thread Group → Add → Timer → Constant Timer
   - Set to `6000` ms (6 seconds between requests)

#### Error: Timeout

**Symptom:** `java.net.SocketTimeoutException`

**Solutions:**
1. Increase timeout in HTTP Request Defaults (Response Timeout > 120000)
2. LLM responses can take 10-30 seconds under load
3. Check server health

### 13.2 JMeter Best Practices

1. **Always use CLI mode for actual load tests** - GUI consumes significant resources
2. **Disable all listeners except file output during load tests**
3. **Use unique session IDs** - Prevents session conflicts
4. **Ramp up gradually** - Don't hit the server with all users instantly
5. **Monitor server resources** - CPU, memory, network during tests
6. **Run multiple shorter tests** - Rather than one very long test

### 13.3 Generating Report from Existing Results

If you have a `.jtl` file and want to generate an HTML report:

```bash
jmeter -g "C:\JMeterTests\results\results.jtl" -o "C:\JMeterTests\results\html_report_new"
```

---

## Appendix A: Complete Test Plan Structure

Your final test plan should look like this:

```
📁 BigTapp Chatbot Load Test (Test Plan)
├── 📋 HTTP Request Defaults
├── 👥 Chatbot Users (Thread Group)
│   ├── 📋 HTTP Header Manager
│   ├── 📋 CSV Data Set Config (Test Messages)
│   ├── ⏱️ Constant Timer (optional)
│   ├── 📤 Health Check (HTTP Request - GET)
│   ├── 📤 Chat API Request (HTTP Request - POST)
│   │   └── ✓ Response Assertion
│   ├── 📤 WhatsApp Webhook (HTTP Request - POST) [optional]
│   │   └── ✓ Response Assertion
│   └── ⏱️ Duration Assertion
├── 📊 View Results Tree
├── 📊 Summary Report
├── 📊 Aggregate Report
└── 📊 Response Time Graph
```

---

## Appendix B: Quick Start Checklist

For a quick load test, complete these steps in order:

1. [ ] Install Java and JMeter
2. [ ] Create new Test Plan → Rename to `BigTapp Chatbot Load Test`
3. [ ] Add `HTTP Request Defaults`:
   - Protocol: `https`
   - Server: `chatbot.bigtapp.com`
   - Port: `443`
4. [ ] Add `Thread Group`:
   - Threads: `20`
   - Ramp-Up: `30`
   - Loop Count: `10`
5. [ ] Add `HTTP Header Manager`:
   - Content-Type: `application/json`
6. [ ] Add `HTTP Request`:
   - Method: `POST`
   - Path: `/agent-chat`
   - Body: `{"session_id": "jmeter_${__UUID}", "message": "I need travel insurance"}`
7. [ ] Add `Response Assertion`:
   - Response Code equals `200`
8. [ ] Add `Summary Report`
9. [ ] Save and run from CLI:
   ```
   jmeter -n -t test.jmx -l results.jtl -e -o report
   ```

---

## Appendix C: Contact & Support

| Contact | Details |
|---------|---------|
| **Application Team** | Contact your IT/DevOps team |
| **JMeter Documentation** | https://jmeter.apache.org/usermanual/ |
| **Server Status** | `curl http://chatbot.bigtapp.com/ready` |

---

*Document Version: 2.0*  
*Last Updated: January 2026*  
*Protocol: HTTP (Port 80)*
