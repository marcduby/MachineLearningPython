


# imports


# constants
variant = "4:g.1801841C>G"

URL_VARIANT_ASSOCIATION = ""


# methods
def convert_variant_to_hugeamp(variant, log=False):
    '''
    converrts to flannick lab notation 
    '''
    # split by period
    list_temp = variant.split(":")
    chrom = list_temp[0]

    list_temp = list_temp[1].split(".")

    list_temp = list_temp[1].split(">")

    alt = list_temp[1]
    ref = list_temp[0][-1]
    pos = list_temp[0][0:-1]



    # return
    return "{}:{}:{}:{}".format(chrom, pos, ref, alt)


def convert_variant_format(variant):
    # Split the variant into chromosome and the rest
    chrom, rest = variant.split(':g.')
    
    # Split the rest into position and the nucleotide change
    position, change = rest[:-1].split('C>') if 'C>' in rest else rest[:-1].split('>G')
    
    # Extract the reference and alternative alleles
    ref = change[0]
    alt = change[2]
    
    # Construct the new format
    new_format = f"{chrom}:{position}:{ref}:{alt}"
    
    return new_format

# # Example usage
# variant = "4:g.1801841C>G"
# new_format = convert_variant_format(variant)
# print(new_format)  # Output: "4:1801841:C:G"




# main
if __name__ == "__main__":
    converted_variant = convert_variant_to_hugeamp(variant=variant)
    print("origina;: {} to new: {}".format(variant, converted_variant))


