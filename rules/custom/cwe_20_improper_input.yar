rule CWE20_ImproperInput_NoValidation
{
    meta:
        cwe         = "CWE-20"
        severity    = "MEDIUM"
        description = "User input used directly without validation"

    strings:
        $input1  = "argv[" ascii
        $input2  = "stdin" ascii
        $input3  = "getenv(" ascii
        $input4  = "fgets(" ascii
        $input5  = "scanf(" ascii
        $valid1  = "validate" nocase ascii
        $valid2  = "sanitize" nocase ascii
        $valid3  = "check" nocase ascii
        $valid4  = "isdigit(" ascii
        $valid5  = "isalpha(" ascii

    condition:
        any of ($input*) and not any of ($valid*)
}

rule CWE20_ImproperInput_FormatString
{
    meta:
        cwe         = "CWE-20"
        severity    = "HIGH"
        description = "User controlled format string"

    strings:
        $printf1 = "printf(" ascii
        $printf2 = "fprintf(" ascii
        $printf3 = "syslog(" ascii
        $input1  = "argv" ascii
        $input2  = "getenv(" ascii
        $safe1   = "\"%s\"" ascii
        $safe2   = "\"%d\"" ascii

    condition:
        any of ($printf*) and any of ($input*) and not any of ($safe*)
}

rule CWE20_FormatString_Direct
{
    meta:
        cwe         = "CWE-20"
        severity    = "HIGH"
        description = "User input used directly as format string"

    strings:
        $printf1 = "printf(" ascii
        $printf2 = "fprintf(" ascii
        $printf3 = "sprintf(" ascii
        $input1  = "argv" ascii
        $input2  = "input" ascii
        $input3  = "user" ascii
        $safe1   = "\"%s\"" ascii
        $safe2   = "\"%d\"" ascii
        $safe3   = "format" ascii

    condition:
        any of ($printf*) and
        any of ($input*) and
        not any of ($safe*)
}