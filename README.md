# Preliminary Requirements

To see this readme in a beautiful way : 
1. If you are on VSCode you can use the "Ctrl+Shift+V" command. 
2. For the others try to visualise it via a markdown visualiser online (for example : https://markdownlivepreview.com/) and copy paste the content of this readme in this website.

It is recommended for this lab to create a **Python virtual environment** before running any code.  
A virtual environment helps you isolate dependencies, ensuring they don’t conflict with system-wide Python packages.

---

## Python Virtual Environment Setup (macOS, Windows, Linux) — *Recommended*

This guide will help you **create**, **activate**, **use**, and **deactivate** Python virtual environments on **macOS**, **Linux**, and **Windows**.

---

### Step 1: Check if Python is Installed

#### macOS / Linux
```bash
python3 --version
```

#### Windows (PowerShell or CMD)
```powershell
python --version
```

If Python is not installed:

- **macOS:** Install via [Homebrew](https://brew.sh)
  ```bash
  brew install python
  ```

- **Linux:** Install using your package manager
  ```bash
  sudo apt install python3 python3-venv
  ```

- **Windows:** Download and install from [python.org/downloads](https://www.python.org/downloads/)

---

### Step 2: Create a Virtual Environment

Make sure you are in the right repository. We advice you to use the unzipped directory of the lab (the project directory). Then run this command :

#### macOS / Linux
```bash
python3 -m venv .venv
```

#### Windows (PowerShell or CMD)
```powershell
python -m venv .venv
```

This will create a new folder named `.venv` containing your isolated Python environment. The . before the name of the folder will allow the directory to not be displayed if you see your tree of files. 

---

### Step 3: Activate the Virtual Environment

#### macOS / Linux
```bash
source .venv/bin/activate
```

#### Windows (PowerShell or CMD)
You will probably see a permission error when activating, so first allow script execution temporarily:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then activate:
```powershell
.\.venv\Scripts\activate
```

Once activated, your terminal prompt should change to show:
```
(.venv)
```
**Don't forget to activate/reactivate your venv everytime you restart your terminal/everytime you are working on the lab.**

---

### Step 4: Installing Required Python Packages

After activating your virtual environment, install all required dependencies from the `requirements.txt` file.

First you may have to upgrade the version of your pip
```bash
pip install --upgrade pip
```

#### macOS / Linux
```bash
pip install -r requirements_maclinux.txt
```

#### Windows (Powershell or CMD)
```bash
pip install -r requirements_windows.txt
```

If this fails : 
You have to install the following packages : 

```bash
pip install numpy
pip install sympy
pip install matplotlib
pip install qiskit
pip install qiskit-aer
pip install 'qiskit[visualization]'
```
---

### Step 5: Deactivating the Virtual Environment

When you are done working in your virtual environment, simply run:
```bash
deactivate
```

Your terminal prompt will return to normal, and Python will use your system’s default interpreter again.

--- 

✅ You now have a working virtual environment to do the lab in. If you are familiar with jupyter notebook please proceed to the lab and good luck with it. For the others here is a tutorial to understand the basics of it.


# Running Jupyter

## Using Jupyter Notebook

You can start Jupyter Notebook via the **command prompt** (Windows) or **terminal** (macOS/Linux) by entering:

```bash
jupyter notebook
```

By default, the current working directory will be the **start-up directory** (Where you venv is setted up). Please make sure that you activated your venv before running this command.

---

### Accessing Jupyter in Your Browser

Once Jupyter Notebook opens in your browser, the URL will look something like:

```
https://localhost:8888/tree
```

* `localhost` is **not a website**; it means the content is being served from your **local machine**.
* Jupyter runs a **local Python server** that serves the notebooks and dashboard to your browser. This makes it platform-independent and allows easier sharing later.

> **Tip:** Even though Jupyter opens in your browser, your notebooks are running locally on your machine. They are **not on the web** until you explicitly share them.

---

### Navigating the Dashboard

The dashboard interface is mostly self-explanatory. To open your lab notebook:

1. Browse to the folder containing `assignment_post-quantum_cryptography.ipynb`.
2. Click on the notebook to open it.
3. In the top-right, select **Python 3 (ipykernel)** as the kernel.

> Each `.ipynb` file is a **notebook**, and creating a new notebook generates a new `.ipynb` file.

---

### Key Concepts: Kernel and Cell

Two important terms in Jupyter Notebook:

#### Kernel

The **kernel** is like the **brain** of the notebook.

* It runs your code, processes it, and returns the output.
* Each notebook is connected to a specific kernel that understands a particular programming language (like Python).

#### Cell

A **cell** is a block or section for code or text.

* You can write code or notes in a cell.
* Running the cell executes the code or renders the text.
* Cells help organize work into small, manageable chunks.

---

### Types of Cells

There are two main types of cells:

1. **Code Cell**

   * Contains code to be executed by the kernel.
   * Output appears directly below the cell.

2. **Markdown Cell**

   * Contains text formatted using **Markdown**.
   * Displays rendered text when the cell is run.

> The first cell in a new notebook defaults to a code cell.

---

### Example: "Hello World!"

1. Create a cell by clicking on the top left "+" symbol. (See you can also modify the type of the cell)

2. Type the following in the first code cell:

```python
print('Hello World!')
```

3. Run the cell:

   * Click the **Run** button in the toolbar, or
   * Press **Ctrl + Enter** on your keyboard.

4. You should see the output displayed directly below the cell:

```
Hello World!
```

✅ You now know the basics of starting Jupyter, navigating the dashboard, understanding kernels and cells, and running your first code! Good luck with the lab!