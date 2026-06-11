import os
import time

def purge_old_exports(tmp_dir="/tmp/mnemos_evaluation_exports/", days=7):
    if not os.path.exists(tmp_dir):
        return 0
        
    cutoff = time.time() - (days * 86400)
    purged_count = 0
    
    for filename in os.listdir(tmp_dir):
        filepath = os.path.join(tmp_dir, filename)
        if os.path.isfile(filepath):
            file_mtime = os.path.getmtime(filepath)
            if file_mtime < cutoff:
                os.remove(filepath)
                purged_count += 1
                
    return purged_count

if __name__ == "__main__":
    count = purge_old_exports()
    print(f"Purged {count} expired sidecar evaluation artifacts.")
