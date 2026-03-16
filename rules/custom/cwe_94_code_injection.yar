rule CWE94_CodeInjection_eval
{
    meta:
        cwe         = "CWE-94"
        severity    = "CRITICAL"
        description = "Dynamic code execution with user input"

    strings:
        $exec1  = "eval(" ascii
        $exec2  = "exec(" ascii
        $exec3  = "system(" ascii
        $exec4  = "popen(" ascii
        $exec5  = "execve(" ascii
        $exec6  = "execvp(" ascii
        $input1 = "argv" ascii
        $input2 = "getenv(" ascii
        $input3 = "fgets(" ascii
        $safe1  = "escapeshellarg(" ascii
        $safe2  = "escapeshellcmd(" ascii

    condition:
        any of ($exec*) and
        any of ($input*) and
        not any of ($safe*)
}

rule CWE94_CodeInjection_DynamicLoad
{
    meta:
        cwe         = "CWE-94"
        severity    = "HIGH"
        description = "Dynamic library loading with user controlled path"

    strings:
        $load1  = "dlopen(" ascii
        $load2  = "LoadLibrary(" ascii
        $input1 = "argv" ascii
        $input2 = "getenv(" ascii
        $input3 = "fgets(" ascii

    condition:
        any of ($load*) and any of ($input*)
}